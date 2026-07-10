# Claude Code Configuration - Hackathon TV5

## Project Overview

This project uses SPARC methodology with Claude-Flow orchestration for hackathon development.

## Build & Test

```bash
npm run build
npm test
npm run lint
npm run typecheck
```

## SPARC Commands

```bash
ruflo sparc modes
ruflo sparc run <mode> "<task>"
ruflo sparc tdd "<feature>"
ruflo sparc batch <modes> "<task>"
ruflo sparc pipeline "<task>"
```

## Agentic QE Fleet

This project includes the Agentic QE Fleet for quality engineering:

- **31 QE Agents**: Test generation, coverage analysis, performance, security
- **Fleet Topology**: hierarchical, max 10 agents
- **Testing Focus**: unit, integration (Jest)

### Integrity Rule

- No shortcuts, no fake data, no false claims
- Use real database queries, not mocks, for integration tests
- Run actual tests, do not assume they pass

### QE Memory Namespace

Agents share state through `aqe/*`:
- `aqe/test-plan/*`, `aqe/coverage/*`, `aqe/quality/*`, `aqe/performance/*`, `aqe/security/*`, `aqe/swarm/coordination`

## Upstream References

See upstream CLAUDE.md files for swarm config, V3 CLI, agent routing, behavioural rules, file organization, security rules, concurrency patterns, and memory commands.
