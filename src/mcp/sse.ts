#!/usr/bin/env node
/**
 * MCP Server - SSE (Server-Sent Events) Transport
 * Run with: npx @agenticsorg/hackathon mcp sse --port 3000
 */

import express from 'express';
import { McpServer } from './server.js';
import type { McpRequest } from '../types.js';

const DEFAULT_PORT = 3000;

export function startSseServer(port: number = DEFAULT_PORT): void {
  const app = express();
  const server = new McpServer();

  app.use(express.json());

  // CORS headers for SSE
  app.use((req, res, next) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    if (req.method === 'OPTIONS') {
      res.sendStatus(200);
      return;
    }
    next();
  });

  // Health check endpoint
  app.get('/health', (req, res) => {
    res.json({ status: 'ok', server: 'agentics-hackathon-mcp' });
  });

  // SSE endpoint for MCP
  app.get('/sse', (req, res) => {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    // Send initial connection event
    res.write(`event: connected\ndata: ${JSON.stringify({ server: 'agentics-hackathon-mcp' })}\n\n`);

    // Keep connection alive
    const keepAlive = setInterval(() => {
      res.write(': keepalive\n\n');
    }, 30000);

    req.on('close', () => {
      clearInterval(keepAlive);
    });
  });

  // JSON-RPC endpoint for MCP requests
  app.post('/rpc', async (req, res) => {
    const request = req.body as McpRequest;

    if (!request.jsonrpc || request.jsonrpc !== '2.0') {
      res.status(400).json({
        jsonrpc: '2.0',
        id: null,
        error: {
          code: -32600,
          message: 'Invalid Request: jsonrpc must be "2.0"'
        }
      });
      return;
    }

    const response = await server.handleRequest(request);
    res.json(response);
  });

  // Info endpoint
  app.get('/', (req, res) => {
    res.json({
      name: 'Agentics Hackathon MCP Server',
      version: '1.0.0',
      transport: 'sse',
      endpoints: {
        sse: '/sse',
        rpc: '/rpc',
        health: '/health'
      },
      capabilities: {
        tools: true,
        resources: true,
        prompts: true
      }
    });
  });

  app.listen(port, () => {
    console.log(`
╔═══════════════════════════════════════════════════════════════╗
║  Agentics Hackathon MCP Server (SSE)                          ║
╠═══════════════════════════════════════════════════════════════╣
║  Status:    Running                                           ║
║  Port:      ${String(port).padEnd(50)}║
║  SSE:       http://localhost:${port}/sse                         ║
║  RPC:       http://localhost:${port}/rpc                         ║
║  Health:    http://localhost:${port}/health                      ║
╚═══════════════════════════════════════════════════════════════╝
    `);
  });
}

// Run if called directly
if (process.argv[1]?.endsWith('sse.js')) {
  const port = parseInt(process.env.PORT || String(DEFAULT_PORT), 10);
  startSseServer(port);
}
