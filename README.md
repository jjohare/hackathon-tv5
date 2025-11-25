# Agentics Foundation TV5 Hackathon

[![npm version](https://img.shields.io/npm/v/@agenticsorg/hackathon.svg)](https://www.npmjs.com/package/@agenticsorg/hackathon)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Node.js](https://img.shields.io/badge/Node.js-18%2B-green.svg)](https://nodejs.org)
[![Discord](https://img.shields.io/discord/1234567890?color=5865F2&logo=discord&logoColor=white&label=Discord)](https://discord.agentics.org)
[![Google Cloud](https://img.shields.io/badge/Powered%20by-Google%20Cloud-4285F4?logo=google-cloud)](https://cloud.google.com)
[![Anthropic](https://img.shields.io/badge/Built%20with-Claude-orange)](https://anthropic.com)

<div align="center">

```
 █████╗  ██████╗ ███████╗███╗   ██╗████████╗██╗ ██████╗███████╗
██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██║██╔════╝██╔════╝
███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ██║██║     ███████╗
██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ██║██║     ╚════██║
██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ██║╚██████╗███████║
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝ ╚═════╝╚══════╝
```

### 🚀 TV5 Hackathon - Supported by Google

**Building the Future of Agentic AI | Open Source | Global**

[Website](https://agentics.org/hackathon) · [Discord](https://discord.agentics.org) · [Documentation](#documentation) · [Get Started](#quick-start)

</div>

---

## 🎯 The Challenge

**Every night, millions spend up to 45 minutes deciding what to watch — billions of hours lost every day. Not from lack of content, but from fragmentation.**

Join the Agentics Foundation TV5 Hackathon to build agentic AI solutions that solve real problems. Powered by Google Cloud, Gemini, Claude, and the best open-source tools in the ecosystem.

## ✨ Features

- 🛠️ **Interactive Setup Wizard** - Get started in minutes with guided project initialization
- 🔧 **Tool Management** - Install and configure AI development tools with a single command
- 🤖 **MCP Server** - Model Context Protocol support for seamless AI integration (STDIO & SSE)
- 🎨 **Beautiful CLI** - Modern terminal UI with colors, spinners, and progress indicators
- 📦 **Zero Config** - Sensible defaults that just work
- 🌐 **Community Integration** - Direct links to Discord and resources

## 🚀 Quick Start

### One-Line Setup

```bash
npx @agenticsorg/hackathon init
```

This will:
1. Check your system prerequisites
2. Guide you through project setup
3. Help you select a hackathon track
4. Install your chosen development tools
5. Connect you with the community

### Global Installation (Optional)

```bash
npm install -g @agenticsorg/hackathon
hackathon init
```

## 📋 Commands

| Command | Description |
|---------|-------------|
| `hackathon init` | Initialize a new hackathon project |
| `hackathon tools` | Browse and install development tools |
| `hackathon status` | Check project configuration status |
| `hackathon info` | View hackathon details and resources |
| `hackathon mcp [stdio\|sse]` | Start the MCP server |
| `hackathon discord` | Join the community Discord |
| `hackathon --help` | Show all available commands |

### Init Options

```bash
hackathon init [options]

Options:
  -f, --force           Force reinitialize existing project
  -y, --yes             Skip prompts, use defaults
  -t, --tools <tools>   Tools to install (space-separated)
  --track <track>       Select hackathon track
  --team <name>         Set team name
```

### Example Workflows

```bash
# Interactive setup (recommended for beginners)
npx @agenticsorg/hackathon init

# Quick setup with specific tools
npx @agenticsorg/hackathon init --tools claudeFlow geminiCli adk

# Check available tools
hackathon tools --list

# Install specific tools later
hackathon tools --install ruvector agentDb

# Start MCP server for AI integration
hackathon mcp sse --port 3000
```

## 🏆 Hackathon Tracks

### 🎬 Entertainment Discovery
Solve the 45-minute decision problem — help users find what to watch across fragmented content platforms.

### 🤝 Multi-Agent Systems
Build collaborative AI agents that work together using Google ADK and Vertex AI.

### ⚡ Agentic Workflows
Create autonomous workflows with Claude, Gemini, and orchestration tools.

### 💡 Open Innovation
Bring your own idea — any agentic AI solution that makes an impact.

## 🔧 Available Tools

### AI Assistants
| Tool | Description | Install |
|------|-------------|---------|
| **Claude Code CLI** | Anthropic's AI coding assistant | `npm i -g @anthropic-ai/claude-code` |
| **Google Gemini CLI** | Google's multimodal AI interface | `npm i -g @google/generative-ai-cli` |

### Orchestration
| Tool | Description | Install |
|------|-------------|---------|
| **Claude Flow** | Multi-agent orchestration framework | `npx claude-flow@alpha init --force` |
| **Google ADK** | Agent Development Kit | `pip install google-adk` |

### Databases
| Tool | Description | Install |
|------|-------------|---------|
| **RuVector** | Vector database & embeddings | `npm install ruvector` |
| **AgentDB** | Agentic AI state management | `npx agentdb init` |
| **Agentic Synth** | Synthesis tools | `npx @ruvector/agentic-synth init` |

### Cloud Platform
| Tool | Description | Install |
|------|-------------|---------|
| **Google Cloud CLI** | Full GCP SDK | [Installation Guide](https://cloud.google.com/sdk/docs/install) |
| **Vertex AI SDK** | ML platform SDK | `pip install google-cloud-aiplatform` |

## 🔌 MCP Server

The hackathon CLI includes a Model Context Protocol (MCP) server for AI integration.

### STDIO Transport

```bash
hackathon mcp stdio
```

Add to your Claude configuration:

```json
{
  "mcpServers": {
    "hackathon": {
      "command": "npx",
      "args": ["@agenticsorg/hackathon", "mcp", "stdio"]
    }
  }
}
```

### SSE Transport

```bash
hackathon mcp sse --port 3000
```

Connect at `http://localhost:3000/sse`

### Available MCP Tools

- `get_hackathon_info` - Get hackathon information
- `get_tracks` - List available tracks
- `get_available_tools` - List development tools
- `get_project_status` - Check project configuration
- `check_tool_installed` - Verify tool installation
- `get_resources` - Get hackathon resources

## 📖 Documentation

- **[Agentics Foundation](https://agentics.org)** - Organization homepage
- **[Hackathon Page](https://agentics.org/hackathon)** - Event details
- **[Google ADK Docs](https://google.github.io/adk-docs/)** - Agent Development Kit
- **[Vertex AI Docs](https://cloud.google.com/vertex-ai/docs)** - Google ML Platform
- **[Claude Docs](https://docs.anthropic.com)** - Anthropic documentation
- **[Gemini API](https://ai.google.dev/gemini-api/docs)** - Google AI documentation

## 🌐 Related Projects

Explore more tools from the ecosystem:

- **[RuVector](https://ruv.io/projects)** - Vector database toolkit
- **[AgentDB](https://ruv.io/projects)** - Agent state management
- **[Agentic Synth](https://ruv.io/projects)** - AI synthesis tools
- **[Claude Flow](https://github.com/anthropics/claude-flow)** - Multi-agent orchestration

## 🤝 Community & Support

<div align="center">

### Join Our Discord Community

[![Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.agentics.org)

**Team Formation | Technical Support | Networking | Announcements**

</div>

- **Discord**: [discord.agentics.org](https://discord.agentics.org)
- **Website**: [agentics.org/hackathon](https://agentics.org/hackathon)
- **GitHub Issues**: [Report a bug](https://github.com/agenticsorg/hackathon-tv5/issues)

## 🏗️ Project Structure

```
your-hackathon-project/
├── .hackathon.json     # Project configuration
├── package.json
├── src/
│   ├── agents/         # Your AI agents
│   ├── workflows/      # Agentic workflows
│   └── index.ts
└── README.md
```

## 📄 Configuration

The CLI creates a `.hackathon.json` file in your project:

```json
{
  "projectName": "my-hackathon-project",
  "teamName": "Awesome Team",
  "track": "multi-agent-systems",
  "tools": {
    "claudeCode": true,
    "claudeFlow": true,
    "geminiCli": true,
    "adk": true
  },
  "mcpEnabled": true,
  "discordLinked": true,
  "initialized": true,
  "createdAt": "2025-01-15T10:00:00.000Z"
}
```

## 🔒 Requirements

- **Node.js** 18.0.0 or higher
- **npm** 9.0.0 or higher
- **Python** 3.9+ (for Google ADK/Vertex AI)
- **Git** (recommended)

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Cloud** - Hackathon sponsor and technology partner
- **Anthropic** - Claude AI and tooling
- **Agentics Foundation** - Organization and community
- **All Contributors** - Thank you for making this possible!

---

<div align="center">

**Built with ❤️ by the [Agentics Foundation](https://agentics.org)**

*Making AI innovation and education open to everyone through open-source agentic AI systems*

[![Website](https://img.shields.io/badge/agentics.org-Visit-blue?style=flat-square)](https://agentics.org)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?style=flat-square&logo=twitter)](https://twitter.com/agenticsorg)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/company/agentics)

</div>
