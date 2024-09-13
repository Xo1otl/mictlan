<?php

$workspaces = [
    "apps/koemade",
];

$dependencies = [
    "packages/util/php"
];

function runComposerInstall($directory)
{
    echo "Running composer install in {$directory}\n";
    chdir($directory);
    passthru('composer install');
    chdir(__DIR__);  // Return to the original directory
}

function runTests($directory)
{
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
        foreach ($workspaces as $workspace) {
            if (is_dir($workspace)) {
                runComposerInstall($workspace);
            } else {
                echo "Warning: Directory {$workspace} not found. Skipping.\n";
            }
        }
        break;
    case 'test':
        foreach ($workspaces + $dependencies as $workspace) {
            if (is_dir($workspace)) {
                runTests($workspace);
            } else {
                echo "Warning: Directory {$workspace} not found. Skipping.\n";
            }
        }
        break;
    default:
        echo "Unknown command: {$command}\n";
        echo "Available commands: install, test\n";
        break;
}

// TODO: file watcher書いてclassmapのautoloadができるようにしてみるのも面白い
// その場合は設定と処理を分離して処理をscriptsに配置する