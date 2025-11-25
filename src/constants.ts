/**
 * Agentics Foundation TV5 Hackathon CLI Constants
 */

import type { Tool, HackathonTrack } from './types.js';

export const HACKATHON_NAME = 'Agentics Foundation TV5 Hackathon';
export const HACKATHON_TAGLINE = 'Building the Future of Agentic AI';
export const DISCORD_URL = 'https://discord.agentics.org';
export const WEBSITE_URL = 'https://agentics.org/hackathon';
export const GITHUB_URL = 'https://github.com/agenticsorg/hackathon-tv5';
export const CONFIG_FILE = '.hackathon.json';

export const TRACKS: Record<HackathonTrack, { name: string; description: string }> = {
  'entertainment-discovery': {
    name: 'Entertainment Discovery',
    description: 'Solve the 45-minute decision problem - help users find what to watch across fragmented content'
  },
  'multi-agent-systems': {
    name: 'Multi-Agent Systems',
    description: 'Build collaborative AI agents that work together using Google ADK and Vertex AI'
  },
  'agentic-workflows': {
    name: 'Agentic Workflows',
    description: 'Create autonomous workflows with Claude, Gemini, and orchestration tools'
  },
  'open-innovation': {
    name: 'Open Innovation',
    description: 'Bring your own idea - any agentic AI solution that makes an impact'
  }
};

export const AVAILABLE_TOOLS: Tool[] = [
  {
    name: 'claudeCode',
    displayName: 'Claude Code CLI',
    description: 'Anthropic\'s official CLI for Claude - AI-powered coding assistant',
    installCommand: 'npm install -g @anthropic-ai/claude-code',
    verifyCommand: 'claude --version',
    docUrl: 'https://docs.anthropic.com/claude-code',
    required: false,
    category: 'ai-assistants'
  },
  {
    name: 'claudeFlow',
    displayName: 'Claude Flow',
    description: 'Multi-agent orchestration framework for Claude',
    installCommand: 'npx claude-flow@alpha init --force',
    verifyCommand: 'npx claude-flow --version',
    docUrl: 'https://github.com/anthropics/claude-flow',
    required: false,
    category: 'orchestration'
  },
  {
    name: 'geminiCli',
    displayName: 'Google Gemini CLI',
    description: 'Command-line interface for Google Gemini models',
    installCommand: 'npm install -g @google/generative-ai-cli',
    verifyCommand: 'gemini --version',
    docUrl: 'https://ai.google.dev/gemini-api/docs',
    required: false,
    category: 'ai-assistants'
  },
  {
    name: 'googleCloudCli',
    displayName: 'Google Cloud CLI (gcloud)',
    description: 'Google Cloud SDK for Vertex AI, Cloud Functions, and more',
    installCommand: 'curl https://sdk.cloud.google.com | bash',
    verifyCommand: 'gcloud --version',
    docUrl: 'https://cloud.google.com/sdk/docs/install',
    required: false,
    category: 'cloud-platform'
  },
  {
    name: 'adk',
    displayName: 'Google Agent Development Kit',
    description: 'Build multi-agent systems with Google\'s ADK',
    installCommand: 'pip install google-adk',
    verifyCommand: 'python -c "import google.adk"',
    docUrl: 'https://google.github.io/adk-docs/',
    required: false,
    category: 'orchestration'
  },
  {
    name: 'vertexAi',
    displayName: 'Vertex AI SDK',
    description: 'Google Cloud\'s unified ML platform SDK',
    installCommand: 'pip install google-cloud-aiplatform',
    verifyCommand: 'python -c "import vertexai"',
    docUrl: 'https://cloud.google.com/vertex-ai/docs',
    required: false,
    category: 'cloud-platform'
  },
  {
    name: 'ruvector',
    displayName: 'RuVector',
    description: 'Vector database and embeddings toolkit for AI applications',
    installCommand: 'npm install ruvector',
    verifyCommand: 'npx ruvector --version',
    docUrl: 'https://ruv.io/projects',
    required: false,
    category: 'databases'
  },
  {
    name: 'agentDb',
    displayName: 'AgentDB',
    description: 'Database designed for agentic AI state management and memory',
    installCommand: 'npx agentdb init',
    verifyCommand: 'npx agentdb --version',
    docUrl: 'https://ruv.io/projects',
    required: false,
    category: 'databases'
  },
  {
    name: 'agenticSynth',
    displayName: 'Agentic Synth',
    description: 'Synthesis tools for agentic AI development',
    installCommand: 'npx @ruvector/agentic-synth init',
    verifyCommand: 'npx @ruvector/agentic-synth --version',
    docUrl: 'https://ruv.io/projects',
    required: false,
    category: 'synthesis'
  }
];

export const BANNER = `
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║     █████╗  ██████╗ ███████╗███╗   ██╗████████╗██╗ ██████╗███████╗           ║
║    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██║██╔════╝██╔════╝           ║
║    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ██║██║     ███████╗           ║
║    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ██║██║     ╚════██║           ║
║    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ██║╚██████╗███████║           ║
║    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝ ╚═════╝╚══════╝           ║
║                                                                               ║
║                    🚀 TV5 HACKATHON - Supported by Google 🚀                  ║
║                                                                               ║
║         Building the Future of Agentic AI | Open Source | Global             ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
`;

export const WELCOME_MESSAGE = `
Welcome to the Agentics Foundation TV5 Hackathon!

Every night, millions spend up to 45 minutes deciding what to watch — billions
of hours lost every day. Not from lack of content, but from fragmentation.

Join us to build the future of agentic AI systems that solve real problems.

🔗 Discord: ${DISCORD_URL}
🌐 Website: ${WEBSITE_URL}
📦 GitHub:  ${GITHUB_URL}
`;
