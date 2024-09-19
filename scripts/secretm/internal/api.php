<?php

require __DIR__ . "/secretm.php";
require __DIR__ . "/cli.php";

function relativePath($path, $allowNonExistent = false)
{
    $workspaceRoot = workspaceFolder();

    // 絶対パスの場合
    if (strpos($path, '/') === 0 || (strlen($path) > 1 && $path[1] === ':')) {
        $absolutePath = $allowNonExistent ? $path : realpath($path);
        if ($absolutePath === false) {
            throw new Exception("The specified absolute path does not exist or is not accessible: $path");
        }
        if (strpos($absolutePath, $workspaceRoot) === 0) {
            return substr($absolutePath, strlen($workspaceRoot) + 1);
        }
        throw new Exception("The specified path is outside the workspace: $path");
    }

    // 相対パスの場合
    $absolutePath = $allowNonExistent ? $workspaceRoot . DIRECTORY_SEPARATOR . $path : realpath($workspaceRoot . DIRECTORY_SEPARATOR . $path);
    if ($absolutePath === false) {
        throw new Exception("The specified relative path does not exist or is not accessible: $path");
    }
    if (strpos($absolutePath, $workspaceRoot) === 0) {
        return substr($absolutePath, strlen($workspaceRoot) + 1);
    }
    throw new Exception("The specified path is outside the workspace: $path");
}

$command = $argv[1] ?? null;
$options = array_slice($argv, 2);

$cli = new Cli();
$manager = new Manager($cli, $cli, $cli);

$handlers = [
    'add' => function ($options) use ($manager) {
        if (empty($options[0])) {
            throw new Exception("Path must be specified for the 'add' command.");
        }
        $relativePath = relativePath($options[0]);
        $manager->add($relativePath);
    },
    'pack' => function ($options) use ($manager) {
        if (empty($options)) {
            $outputPath = workspaceFolder();
        } elseif (count($options) == 1) {
            $outputPath = $options[0];
        }
        $relativePath = relativePath($outputPath);
        $archivePath = ($relativePath === '') ? 'secrets.tar.gz' : $relativePath . DIRECTORY_SEPARATOR . 'secrets.tar.gz';
        $manager->pack($archivePath);
    },
    'unpack' => function ($options) use ($manager) {
        if (empty($options)) {
            $archivepath = workspaceFolder();
        } elseif (count($options) == 1) {
            $archivepath = $options[0];
        }
        $archivePath = relativePath($archivepath . DIRECTORY_SEPARATOR . "secrets.tar.gz");
        $manager->tar->decompress($archivePath);
    },
];

if (array_key_exists($command, $handlers)) {
    try {
        $handlers[$command]($options);
    } catch (Exception $e) {
        echo "Error: " . $e->getMessage() . "\n";
        exit(1);
    }
} else {
    echo "Usage: php " . basename(__FILE__) . " <command> [options]\n\n";
    echo "Available commands:\n";
    echo "  add <path>    : Add a secret file or directory to the manager\n";
    echo "  pack [dir]    : Pack all secrets into an archive\n";
    echo "  unpack [file] : Unpack secrets from an archive\n\n";
    echo "Detailed usage:\n\n";
    echo "1. Project setup:\n";
    echo "   - Place a 'workspace.php' file in your project's root directory.\n";
    echo "   - This file is used to identify the workspace root.\n\n";
    echo "2. Path specifications:\n";
    echo "   - Use absolute paths or paths relative to the workspace root.\n";
    echo "   - The workspace root is the directory containing 'workspace.php'.\n\n";
    echo "3. Adding secrets (add command):\n";
    echo "   - Use: php " . basename(__FILE__) . " add <path>\n";
    echo "   - <path> can be a file or directory.\n";
    echo "   - Added secrets are listed in 'build/secrets.json'.\n\n";
    echo "4. Packing secrets (pack command):\n";
    echo "   - Use: php " . basename(__FILE__) . " pack [dir]\n";
    echo "   - [dir] is optional. If omitted, packs to the workspace root.\n";
    echo "   - Creates 'secrets.tar.gz' in the specified directory.\n";
    echo "   - Run this after adding all desired secrets.\n\n";
    echo "5. Unpacking secrets (unpack command):\n";
    echo "   - Use: php " . basename(__FILE__) . " unpack [file]\n";
    echo "   - [file] is optional. Default is 'secrets.tar.gz' in the workspace root.\n";
    echo "   - Extracts secrets to their original locations within the workspace.\n\n";
    echo "6. Workflow:\n";
    echo "   1) Add secrets:    php " . basename(__FILE__) . " add path/to/secret\n";
    echo "   2) Repeat step 1 for all secrets\n";
    echo "   3) Pack secrets:   php " . basename(__FILE__) . " pack [output_dir]\n";
    echo "   4) (Later) Unpack: php " . basename(__FILE__) . " unpack [secrets_file]\n\n";
    echo "Note: Always ensure you're running the script from within the workspace.\n";
}
