# @agenticsorg/hackathon

> CLI and MCP server for the Agentics Foundation TV5 Hackathon - Build the future of agentic AI with Google Cloud, Gemini, Claude, and open-source tools

[![npm version](https://img.shields.io/npm/v/@agenticsorg/hackathon.svg)](https://www.npmjs.com/package/@agenticsorg/hackathon)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-5865F2)](https://discord.agentics.org)

## Quick Start

```bash
npx @agenticsorg/hackathon init
```

## The Challenge

**Every night, millions spend up to 45 minutes deciding what to watch.** Join us to build agentic AI solutions powered by Google Cloud, Gemini, and Claude.

## Features

- 🛠️ Interactive setup wizard
- 🔧 One-command tool installation
- 🤖 MCP server (STDIO & SSE)
- 🎨 Beautiful terminal UI
- 🌐 Community integration

## Commands

```bash
hackathon init          # Initialize project
hackathon tools         # Manage dev tools
hackathon status        # Check configuration
hackathon info          # View hackathon details
hackathon mcp [mode]    # Start MCP server
hackathon discord       # Join community
```

## Hackathon Tracks

- 🎬 **Entertainment Discovery** - Content recommendation
- 🤝 **Multi-Agent Systems** - Collaborative AI
- ⚡ **Agentic Workflows** - Autonomous workflows
- 💡 **Open Innovation** - Your own idea

## Available Tools

| Tool | Install |
|------|---------|
| Claude Code CLI | `npm i -g @anthropic-ai/claude-code` |
| Claude Flow | `npx claude-flow@alpha init --force` |
| Google Gemini CLI | `npm i -g @google/generative-ai-cli` |
| Google ADK | `pip install google-adk` |
| RuVector | `npm install ruvector` |
| AgentDB | `npx agentdb init` |
| Agentic Synth | `npx @ruvector/agentic-synth init` |

## MCP Integration

```bash
# STDIO
hackathon mcp stdio

# SSE
hackathon mcp sse --port 3000
```

Add to Claude configuration:
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

## Resources

- 🌐 [Website](https://agentics.org/hackathon)
- 💬 [Discord](https://discord.agentics.org)
- 📦 [GitHub](https://github.com/agenticsorg/hackathon-tv5)
- 📚 [Google ADK Docs](https://google.github.io/adk-docs/)

## Requirements

- Node.js 18+
- npm 9+
- Python 3.9+ (optional)

## License

Apache-2.0

---

**Built with ❤️ by [Agentics Foundation](https://agentics.org)**
