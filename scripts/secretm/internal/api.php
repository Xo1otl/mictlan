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
    echo "Available commands: " . implode(', ', array_keys($handlers)) . "\n";
}