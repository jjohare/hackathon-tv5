/**
 * Status command - Show current hackathon project status
 */

import chalk from 'chalk';
import { AVAILABLE_TOOLS, TRACKS, DISCORD_URL, WEBSITE_URL } from '../constants.js';
import { logger, loadConfig, configExists, checkToolInstalled } from '../utils/index.js';

export async function statusCommand(): Promise<void> {
  if (!configExists()) {
    logger.warning('Not initialized. Run `hackathon init` first.');
    return;
  }

  const config = loadConfig();
  if (!config) {
    logger.error('Failed to load configuration');
    return;
  }

  logger.divider();
  console.log(chalk.bold.cyan('  Hackathon Project Status'));
  logger.divider();
  logger.newline();

  // Project info
  logger.table({
    'Project': config.projectName,
    'Team': config.teamName || 'Not set',
    'Track': config.track ? TRACKS[config.track].name : 'Not selected',
    'Initialized': new Date(config.createdAt).toLocaleDateString(),
    'MCP Server': config.mcpEnabled ? 'Enabled' : 'Disabled',
    'Discord': config.discordLinked ? 'Connected' : 'Not connected'
  });

  logger.newline();
  logger.divider();
  console.log(chalk.bold.cyan('  Tools Status'));
  logger.divider();
  logger.newline();

  // Check tools
  const enabledTools = Object.entries(config.tools)
    .filter(([_, enabled]) => enabled)
    .map(([name]) => name);

  if (enabledTools.length === 0) {
    logger.info('No tools configured. Run `hackathon tools` to install tools.');
  } else {
    for (const toolName of enabledTools) {
      const tool = AVAILABLE_TOOLS.find(t => t.name === toolName);
      if (tool) {
        const installed = await checkToolInstalled(tool);
        const status = installed ? chalk.green('✔ Ready') : chalk.yellow('⚠ Needs setup');
        console.log(`  ${status} ${tool.displayName}`);
      }
    }
  }

  logger.newline();
  logger.divider();
  console.log(chalk.bold.cyan('  Resources'));
  logger.divider();
  logger.newline();

  logger.table({
    'Website': WEBSITE_URL,
    'Discord': DISCORD_URL,
    'Config File': '.hackathon.json'
  });

  logger.newline();
}
