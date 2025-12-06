/**
 * Tool installation utilities
 */

import { spawn, exec } from 'child_process';
import { promisify } from 'util';
import ora from 'ora';
import type { Tool, InstallProgress } from '../types.js';
import { logger } from './logger.js';

const execAsync = promisify(exec);

export async function checkToolInstalled(tool: Tool): Promise<boolean> {
  try {
    await execAsync(tool.verifyCommand);
    return true;
  } catch {
    return false;
  }
}

export async function installTool(tool: Tool): Promise<InstallProgress> {
  const spinner = ora(`Installing ${tool.displayName}...`).start();

  try {
    // Handle different install command types
    const command = tool.installCommand;

    if (command.startsWith('npx ')) {
      // For npx commands, just run them directly
      await runCommand(command);
    } else if (command.startsWith('npm install')) {
      await runCommand(command);
    } else if (command.startsWith('pip install')) {
      await runCommand(command);
    } else if (command.startsWith('curl')) {
      // Special handling for curl-based installs (like gcloud)
      spinner.info(`${tool.displayName} requires manual installation`);
      spinner.stop();
      logger.info(`Run: ${command}`);
      logger.link('Installation guide', tool.docUrl);
      return {
        tool: tool.name,
        status: 'skipped',
        message: 'Requires manual installation'
      };
    } else {
      await runCommand(command);
    }

    // Verify installation
    const installed = await checkToolInstalled(tool);

    if (installed) {
      spinner.succeed(`${tool.displayName} installed successfully`);
      return { tool: tool.name, status: 'success' };
    } else {
      spinner.warn(`${tool.displayName} may need additional setup`);
      return {
        tool: tool.name,
        status: 'success',
        message: 'Installed but verification pending'
      };
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    spinner.fail(`Failed to install ${tool.displayName}`);
    logger.error(message);
    return { tool: tool.name, status: 'failed', message };
  }
}

export async function runCommand(command: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const parts = command.split(' ');
    const cmd = parts[0];
    const args = parts.slice(1);

    const child = spawn(cmd, args, {
      shell: true,
      stdio: 'pipe'
    });

    let stdout = '';
    let stderr = '';

    child.stdout?.on('data', (data) => {
      stdout += data.toString();
    });

    child.stderr?.on('data', (data) => {
      stderr += data.toString();
    });

    child.on('close', (code) => {
      if (code === 0) {
        resolve(stdout);
      } else {
        reject(new Error(stderr || `Command exited with code ${code}`));
      }
    });

    child.on('error', reject);
  });
}

export async function checkPrerequisites(): Promise<{
  node: boolean;
  npm: boolean;
  python: boolean;
  pip: boolean;
  git: boolean;
}> {
  const checks = {
    node: false,
    npm: false,
    python: false,
    pip: false,
    git: false
  };

  try {
    await execAsync('node --version');
    checks.node = true;
  } catch { /* not installed */ }

  try {
    await execAsync('npm --version');
    checks.npm = true;
  } catch { /* not installed */ }

  try {
    await execAsync('python3 --version');
    checks.python = true;
  } catch {
    try {
      await execAsync('python --version');
      checks.python = true;
    } catch { /* not installed */ }
  }

  try {
    await execAsync('pip3 --version');
    checks.pip = true;
  } catch {
    try {
      await execAsync('pip --version');
      checks.pip = true;
    } catch { /* not installed */ }
  }

  try {
    await execAsync('git --version');
    checks.git = true;
  } catch { /* not installed */ }

  return checks;
}
