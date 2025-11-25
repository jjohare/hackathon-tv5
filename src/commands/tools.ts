/**
 * Tools command - List and install hackathon tools
 */

import { prompt } from 'enquirer';
import chalk from 'chalk';
import type { Tool } from '../types.js';
import { AVAILABLE_TOOLS } from '../constants.js';
import {
  logger,
  loadConfig,
  updateConfig,
  checkToolInstalled,
  installTool
} from '../utils/index.js';

interface ToolsOptions {
  install?: string[];
  list?: boolean;
  check?: boolean;
}

export async function toolsCommand(options: ToolsOptions): Promise<void> {
  if (options.list || (!options.install && !options.check)) {
    await listTools();
    return;
  }

  if (options.check) {
    await checkTools();
    return;
  }

  if (options.install && options.install.length > 0) {
    await installTools(options.install);
    return;
  }

  // Interactive mode
  await interactiveToolInstall();
}

async function listTools(): Promise<void> {
  logger.info('Available tools for the hackathon:\n');

  const categories = {
    'ai-assistants': 'AI Assistants',
    'orchestration': 'Orchestration & Agent Frameworks',
    'cloud-platform': 'Cloud Platform',
    'databases': 'Databases & Memory',
    'synthesis': 'Synthesis & Advanced Tools',
    'python-frameworks': 'Python Frameworks'
  };

  for (const [category, label] of Object.entries(categories)) {
    const tools = AVAILABLE_TOOLS.filter(t => t.category === category);
    if (tools.length > 0) {
      console.log(chalk.bold.cyan(`\n${label}:`));
      for (const tool of tools) {
        const installed = await checkToolInstalled(tool);
        const status = installed ? chalk.green('✔') : chalk.gray('○');
        console.log(`  ${status} ${chalk.bold(tool.displayName)}`);
        console.log(`    ${chalk.gray(tool.description)}`);
        console.log(`    ${chalk.gray('Install:')} ${chalk.cyan(tool.installCommand)}`);
      }
    }
  }

  logger.newline();
  logger.info('Run `hackathon tools --install <tool>` to install a specific tool');
  logger.info('Run `hackathon tools --check` to check installed status');
}

async function checkTools(): Promise<void> {
  logger.info('Checking installed tools...\n');

  const results: { tool: Tool; installed: boolean }[] = [];

  for (const tool of AVAILABLE_TOOLS) {
    const installed = await checkToolInstalled(tool);
    results.push({ tool, installed });
  }

  const installed = results.filter(r => r.installed);
  const notInstalled = results.filter(r => !r.installed);

  if (installed.length > 0) {
    console.log(chalk.bold.green('Installed:'));
    installed.forEach(({ tool }) => {
      console.log(`  ${chalk.green('✔')} ${tool.displayName}`);
    });
  }

  if (notInstalled.length > 0) {
    console.log(chalk.bold.yellow('\nNot Installed:'));
    notInstalled.forEach(({ tool }) => {
      console.log(`  ${chalk.gray('○')} ${tool.displayName}`);
    });
  }

  logger.newline();
  logger.info(`${installed.length}/${results.length} tools installed`);
}

async function installTools(toolNames: string[]): Promise<void> {
  const config = loadConfig();

  for (const name of toolNames) {
    const tool = AVAILABLE_TOOLS.find(
      t => t.name === name || t.displayName.toLowerCase() === name.toLowerCase()
    );

    if (!tool) {
      logger.error(`Unknown tool: ${name}`);
      logger.info('Run `hackathon tools --list` to see available tools');
      continue;
    }

    const result = await installTool(tool);

    if (result.status === 'success' && config) {
      const toolKey = tool.name as keyof typeof config.tools;
      updateConfig({
        tools: { ...config.tools, [toolKey]: true }
      });
    }
  }
}

async function interactiveToolInstall(): Promise<void> {
  const choices = AVAILABLE_TOOLS.map(tool => ({
    name: tool.name,
    message: tool.displayName,
    hint: tool.description
  }));

  const { selectedTools } = await prompt<{ selectedTools: string[] }>({
    type: 'multiselect',
    name: 'selectedTools',
    message: 'Select tools to install:',
    choices
  });

  if (selectedTools.length === 0) {
    logger.info('No tools selected');
    return;
  }

  await installTools(selectedTools);
}
