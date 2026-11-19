
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * MaintainIQ - AI Service
 * Gemini-powered maintenance intelligence and recommendations.
 */

import { GoogleGenAI, Type } from '@google/genai';

import Issue from '../models/Issue.js';
import Asset from '../models/Asset.js';
import User from '../models/User.js';
import Maintenance from '../models/Maintenance.js';

// ============================================================
// 1. GEMINI CONFIGURATION
// ============================================================

const GEMINI_MODEL =
  process.env.GEMINI_MODEL || 'gemini-3.8-flash';

let aiClient = null;

/**
 * Get Gemini client only when API key is available.
 */
const getAIClient = () => {
  const apiKey = process.env.GEMINI_API_KEY?.trim();

  if (!apiKey) {
    return null;
  }

  if (!aiClient) {
    aiClient = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'MaintainIQ',
        },
      },
    });
  }

  return aiClient;
};

const hasAIKey = () => {
  return Boolean(process.env.GEMINI_API_KEY?.trim());
};

// ============================================================
// 2. ERROR HANDLING
// ============================================================

const handleAIError = (context, err) => {
  const status = err?.status || err?.code;
  const message = err?.message || 'Unknown Gemini API error';

  if (
    status === 429 ||
    message.includes('quota') ||
    message.includes('RESOURCE_EXHAUSTED')
  ) {
    console.warn(
      `⚠️ Gemini quota/rate limit exceeded in ${context}. Using fallback.`
    );
    return;
  }

  if (status === 404) {
    console.error(
      `❌ Gemini model or endpoint unavailable in ${context}.`,
      message
    );
    return;
  }

  if (status === 401 || status === 403) {
    console.error(
      `❌ Gemini authentication/permission error in ${context}.`,
      message
    );
    return;
  }

  console.error(`❌ ${context} Error:`, message);
};

// ============================================================
// 3. RESPONSE HELPERS
// ============================================================

const getResponseText = (response) => {
  const text = response?.text;

  if (typeof text !== 'string' || !text.trim()) {
    throw new Error('Gemini returned an empty response.');
  }

  return text.trim();
};

const parseAIJson = (response) => {
  let text = getResponseText(response);

  // Remove Markdown JSON code fences if present.
  text = text
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/, '')
    .trim();

  try {
    return JSON.parse(text);
  } catch {
    throw new Error('Gemini returned invalid JSON.');
  }
};

const generateAIContent = async (contents, config = {}) => {
  const client = getAIClient();

  if (!client) {
    throw new Error('GEMINI_API_KEY is not configured.');
  }

  return client.models.generateContent({
    model: GEMINI_MODEL,
    contents,
    config,
  });
};

// ============================================================
// 4. FALLBACK RESPONSES
// ============================================================

const getMaintenanceSummaryFallback = (notes) => {
  return `[ENGINEERING AUTO-REPORT]

Maintenance inspection completed.

Technician Notes:
${notes}

The reported observations have been recorded for maintenance review. Further verification should be performed according to the applicable maintenance procedure.`;
};

const getPreventiveRecommendationFallback = (asset) => {
  return {
    suggestedNextMaintenance:
      asset.nextService || 'Next Quarter',

    suggestedReplacement:
      asset.condition === 'Poor'
        ? 'Replacement assessment recommended within the next 3 months.'
        : 'No immediate replacement recommendation. Continue monitoring.',

    inspectionSchedule:
      'Every 30 days - standard visual inspection.',

    narrative:
      'Perform routine inspections, clean operational surfaces, check critical components and verify equipment performance.',
  };
};

const getRepairEstimateFallback = () => ({
  estimatedTimeMinutes: 90,
  estimatedCostMin: 100,
  estimatedCostMax: 250,
  confidence: 'Low',
});

// ============================================================
// 5. AI SERVICE
// ============================================================

export const aiService = {

  // ----------------------------------------------------------
  // FEATURE 1: AI MAINTENANCE SUMMARY
  // ----------------------------------------------------------

  generateMaintenanceSummary: async (inspectionNotes) => {
    if (
      !inspectionNotes ||
      typeof inspectionNotes !== 'string' ||
      inspectionNotes.trim().length < 5
    ) {
      throw new Error(
        'Inspection notes are too short for generating an AI summary.'
      );
    }

    const notes = inspectionNotes.trim();

    if (!hasAIKey()) {
      return getMaintenanceSummaryFallback(notes);
    }

    try {
      const prompt = `
Convert the following rough maintenance notes into a formal,
professional engineering maintenance report.

Include:
- Work performed
- Observations
- Issues identified
- Corrective actions
- Recommended follow-up

Do not invent measurements, repairs or inspection results.

Technician Notes:
${notes}
`;

      const response = await generateAIContent(prompt, {
        systemInstruction:
          'You are an expert facilities management and maintenance engineer. Generate accurate, professional maintenance reports. Never invent technical facts.',
      });

      return getResponseText(response);

    } catch (err) {
      handleAIError('generateMaintenanceSummary', err);
      return getMaintenanceSummaryFallback(notes);
    }
  },

  // ----------------------------------------------------------
  // FEATURE 2: AI PREVENTIVE MAINTENANCE RECOMMENDATION
  // ----------------------------------------------------------

  getPreventiveRecommendation: async (assetId) => {
    try {
      const asset = await Asset.findById(assetId);

      if (!asset) {
        throw new Error('Asset not found.');
      }

      const [issues, maintenances] = await Promise.all([
        Issue.find({ assetId }),
        Maintenance.find({ assetId }),
      ]);

      const assetSummary = `
Asset Name: ${asset.assetName}
Category: ${asset.category}
Location: ${asset.location}
Current Condition: ${asset.condition}
Current Status: ${asset.status}
Next Service: ${asset.nextService || 'Not specified'}

Total Logged Issues: ${issues.length}
Total Maintenance Records: ${maintenances.length}

Recent Issues:
${issues.slice(0, 10).map((issue) => `
- Title: ${issue.title || 'N/A'}
- Category: ${issue.category || 'N/A'}
- Priority: ${issue.priority || 'N/A'}
- Status: ${issue.status || 'N/A'}
`).join('\n')}
`;

      if (!hasAIKey()) {
        return getPreventiveRecommendationFallback(asset);
      }

      const prompt = `
Analyze the following equipment information and generate
a practical preventive maintenance recommendation.

${assetSummary}

Return recommendations based only on available information.
Do not invent inspection results or equipment specifications.
`;

      const response = await generateAIContent(prompt, {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            suggestedNextMaintenance: {
              type: Type.STRING,
              description:
                'Suggested maintenance date or interval.',
            },
            suggestedReplacement: {
              type: Type.STRING,
              description:
                'Actionable equipment or parts replacement recommendation.',
            },
            inspectionSchedule: {
              type: Type.STRING,
              description:
                'Recommended recurring inspection schedule.',
            },
            narrative: {
              type: Type.STRING,
              description:
                'Detailed preventive maintenance recommendation.',
            },
          },
          required: [
            'suggestedNextMaintenance',
            'suggestedReplacement',
            'inspectionSchedule',
            'narrative',
          ],
        },
      });

      return parseAIJson(response);

    } catch (err) {
      handleAIError('getPreventiveRecommendation', err);

      // Preserve actual database errors rather than hiding them.
      if (err.message === 'Asset not found.') {
        throw err;
      }

      return {
        suggestedNextMaintenance: 'Standard schedule',
        suggestedReplacement: 'Inspect during next routine maintenance.',
        inspectionSchedule: 'Monthly inspection',
        narrative:
          'AI recommendations are temporarily unavailable. Follow the approved maintenance schedule and inspect critical equipment components.',
      };
    }
  },

  // ----------------------------------------------------------
  // FEATURE 3: AI SIMILAR ISSUE FINDER
  // ----------------------------------------------------------

  findSimilarIssues: async (description, category) => {
    try {
      if (
        !description ||
        typeof description !== 'string' ||
        !description.trim()
      ) {
        return [];
      }

      const query = {
        status: { $in: ['Resolved', 'Closed'] },
      };

      if (category) {
        query.category = category;
      }

      const historicalIssues = await Issue.find(query).limit(10);

      if (historicalIssues.length === 0) {
        return [];
      }

      if (!hasAIKey()) {
        const firstMatch = historicalIssues[0];

        return [{
          issueNumber: firstMatch.issueNumber || 'N/A',
          title: firstMatch.title || 'Historical issue',
          similarityScore: 85,
          recommendedAction:
            'Review the historical repair notes and verify the affected components.',
        }];
      }

      const formattedHistory = historicalIssues.map((issue) => ({
        issueNumber: issue.issueNumber || 'N/A',
        title: issue.title || 'N/A',
        description: issue.description || 'N/A',
        category: issue.category || 'N/A',
      }));

      const prompt = `
Compare this newly reported issue against the historical resolved issues.

New Issue:
${description}

Category:
${category || 'Not specified'}

Historical Issues:
${JSON.stringify(formattedHistory, null, 2)}

Return the most relevant matches.
Only recommend actions supported by the available historical information.
Return an empty array if there are no meaningful matches.
`;

      const response = await generateAIContent(prompt, {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              issueNumber: {
                type: Type.STRING,
              },
              title: {
                type: Type.STRING,
              },
              similarityScore: {
                type: Type.INTEGER,
                description: 'Similarity percentage from 0 to 100.',
              },
              recommendedAction: {
                type: Type.STRING,
              },
            },
            required: [
              'issueNumber',
              'title',
              'similarityScore',
              'recommendedAction',
            ],
          },
        },
      });

      const results = parseAIJson(response);

      if (!Array.isArray(results)) {
        throw new Error('Invalid similar issue response format.');
      }

      return results;

    } catch (err) {
      handleAIError('findSimilarIssues', err);
      return [];
    }
  },

  // ----------------------------------------------------------
  // FEATURE 4: ASSET HEALTH SCORE
  // ----------------------------------------------------------

  calculateAssetHealthScore: async (assetId) => {
    try {
      const asset = await Asset.findById(assetId);

      if (!asset) {
        return 100;
      }

      const [
        totalIssues,
        criticalIssues,
        unresolvedIssues,
      ] = await Promise.all([
        Issue.countDocuments({ assetId }),

        Issue.countDocuments({
          assetId,
          priority: { $in: ['High', 'Critical'] },
        }),

        Issue.countDocuments({
          assetId,
          status: { $nin: ['Resolved', 'Closed'] },
        }),
      ]);

      let baseScore = 100;

      if (asset.condition === 'Good') {
        baseScore -= 10;
      }

      if (asset.condition === 'Fair') {
        baseScore -= 25;
      }

      if (asset.condition === 'Poor') {
        baseScore -= 50;
      }

      baseScore -= totalIssues * 5;
      baseScore -= criticalIssues * 15;
      baseScore -= unresolvedIssues * 10;

      return Math.max(0, Math.min(100, baseScore));

    } catch (err) {
      console.error(
        '❌ Calculate Asset Health Score Error:',
        err.message
      );

      return 100;
    }
  },

  // ----------------------------------------------------------
  // FEATURE 5: ASSET RISK ASSESSMENT
  // ----------------------------------------------------------

  calculateRiskAssessment: async (assetId) => {
    try {
      const asset = await Asset.findById(assetId);

      if (!asset) {
        return 'Low';
      }

      const [criticalActive, highActive, totalActive] =
        await Promise.all([
          Issue.countDocuments({
            assetId,
            priority: 'Critical',
            status: { $nin: ['Resolved', 'Closed'] },
          }),

          Issue.countDocuments({
            assetId,
            priority: 'High',
            status: { $nin: ['Resolved', 'Closed'] },
          }),

          Issue.countDocuments({
            assetId,
            status: { $nin: ['Resolved', 'Closed'] },
          }),
        ]);

      if (criticalActive > 0 || asset.condition === 'Poor') {
        return 'Critical';
      }

      if (highActive > 0 || asset.condition === 'Fair') {
        return 'High';
      }

      if (totalActive > 1) {
        return 'Medium';
      }

      return 'Low';

    } catch (err) {
      console.error(
        '❌ AI Risk Assessment Error:',
        err.message
      );

      return 'Low';
    }
  },

  // ----------------------------------------------------------
  // FEATURE 6 & 7: REPAIR TIME AND COST ESTIMATION
  // ----------------------------------------------------------

  estimateRepairTimeAndCost: async (description, category) => {
    try {
      if (!hasAIKey()) {
        return getRepairEstimateFallback();
      }

      if (!description || !description.trim()) {
        throw new Error('Issue description is required.');
      }

      const prompt = `
Estimate the repair duration and approximate repair cost
for the following maintenance issue.

Category: ${category || 'Not specified'}
Description: ${description}

Cost must be estimated in USD.
Return reasonable estimates and a confidence level.
Do not present estimates as confirmed quotations.
`;

      const response = await generateAIContent(prompt, {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            estimatedTimeMinutes: {
              type: Type.INTEGER,
              description: 'Estimated repair duration in minutes.',
            },
            estimatedCostMin: {
              type: Type.INTEGER,
              description: 'Minimum estimated cost in USD.',
            },
            estimatedCostMax: {
              type: Type.INTEGER,
              description: 'Maximum estimated cost in USD.',
            },
            confidence: {
              type: Type.STRING,
              description: 'Low, Medium, or High.',
            },
          },
          required: [
            'estimatedTimeMinutes',
            'estimatedCostMin',
            'estimatedCostMax',
            'confidence',
          ],
        },
      });

      const result = parseAIJson(response);

      if (
        !Number.isFinite(result.estimatedTimeMinutes) ||
        !Number.isFinite(result.estimatedCostMin) ||
        !Number.isFinite(result.estimatedCostMax) ||
        result.estimatedTimeMinutes < 0 ||
        result.estimatedCostMin < 0 ||
        result.estimatedCostMax < result.estimatedCostMin
      ) {
        throw new Error('Gemini returned invalid repair estimates.');
      }

      return result;

    } catch (err) {
      handleAIError('estimateRepairTimeAndCost', err);
      return getRepairEstimateFallback();
    }
  },

  // ----------------------------------------------------------
  // FEATURE 8: TECHNICIAN RECOMMENDATION
  // ----------------------------------------------------------

  recommendTechnician: async (issueCategory) => {
    try {
      const technicians = await User.find({
        role: 'Technician',
      });

      if (technicians.length === 0) {
        return null;
      }

      const matchingIssues = await Issue.find({
        category: issueCategory,
        status: 'Resolved',
        assignedTechnician: { $ne: null },
      });

      const techStats = {};

      technicians.forEach((technician) => {
        const id = technician._id.toString();

        techStats[id] = {
          id: technician._id,
          name: technician.name,
          matchCount: 0,
        };
      });

      matchingIssues.forEach((issue) => {
        if (!issue.assignedTechnician) {
          return;
        }

        const technicianId =
          issue.assignedTechnician.toString();

        if (techStats[technicianId]) {
          techStats[technicianId].matchCount += 1;
        }
      });

      const recommendations = Object.values(techStats).sort(
        (a, b) => b.matchCount - a.matchCount
      );

      const topTechnician = recommendations[0];

      if (!topTechnician) {
        return null;
      }

      return {
        id: topTechnician.id,
        name: topTechnician.name,

        confidenceScore:
          topTechnician.matchCount > 0 ? 95 : 75,

        reason:
          topTechnician.matchCount > 0
            ? `Highest number of resolved cases (${topTechnician.matchCount}) under category ${issueCategory}.`
            : `Suggested based on technician availability for category ${issueCategory}.`,
      };

    } catch (err) {
      console.error(
        '❌ Technician Recommendation Error:',
        err.message
      );

      return null;
    }
  },
};
