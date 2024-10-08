<?php

require __DIR__ . "/../vendor/autoload.php";

use util\workspace;

$workspace = new workspace\PHP();
$workspaceFolder = $workspace->root();
$packages = $workspace->packages();

function runComposerInstall($directory)
{
    global $workspaceFolder;
    $directory = $workspaceFolder . DIRECTORY_SEPARATOR . $directory;
    if (!file_exists("$directory/composer.json")) {
        echo "Error: composer.json not found in {$directory}. Skipping.\n";
        return;
    }
    echo "Running composer install in {$directory}\n";
    chdir($directory);
    passthru('composer install');
    chdir(__DIR__);  // Return to the original directory
}

function runTests($directory)
{
    global $workspaceFolder;
    $directory = $workspaceFolder . "/" . $directory;
    if (!file_exists("$directory/composer.json")) {
        echo "Error: composer.json not found in {$directory}. Skipping.\n";
        return;
    }
    echo "Running tests in {$directory}\n";
    $iterator = new RecursiveIteratorIterator(new RecursiveDirectoryIterator($directory));
    $testFiles = new RegexIterator($iterator, '/^.+\.test\.php$/i', RecursiveRegexIterator::GET_MATCH);

    foreach ($testFiles as $file) {
        $filePath = $file[0];
        echo "Executing test file: {$filePath}\n";
        passthru("php {$filePath}");
    }
}

if ($argc < 2) {
    echo "Usage: php workspace.php <command>\n";
    echo "Available commands: install, test\n";
    exit(1);
}

$command = $argv[1];

switch ($command) {
    case 'install':
        foreach ($packages as $package) {
            runComposerInstall($package);
        }
        break;
    case 'test':
        foreach ($packages as $package) {
            runTests($package);
        }
        break;
    default:
        echo "Unknown command: {$command}\n";
        echo "Available commands: install, test\n";
        break;
}

// TODO: file watcher書いてclassmapのautoloadができるようにしてみるのも面白い
// その場合は設定と処理を分離して処理をscriptsに配置する
