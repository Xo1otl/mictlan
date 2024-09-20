<?php

namespace secretm;

require __DIR__ . "/../vendor/autoload.php";

use util\workspace;

$command = $argv[1] ?? null;
$options = array_slice($argv, 2);

$workspace = new workspace\PHP();
$cli = new Cli($workspace);
$manager = new Manager($cli, $cli, $cli);

$handlers = [
    'add' => function ($options) use ($manager, $workspace) {
        if (empty($options[0])) {
            throw new \Exception("Path must be specified for the 'add' command.");
        }
        $relativePath = $workspace->locate($options[0]);
        $manager->add($relativePath);
    },
    'export' => function ($options) use ($manager, $workspace) {
        if (empty($options)) {
            $outputPath = $workspace->locate("");
        } elseif (count($options) == 1) {
            $outputPath = $options[0];
        }
        $relativePath = $workspace->locate($outputPath);
        $archivePath = ($relativePath === '') ? 'secrets.tar.gz' : $relativePath . DIRECTORY_SEPARATOR . 'secrets.tar.gz';
        $manager->export($archivePath);
    },
    'import' => function ($options) use ($manager, $workspace) {
        if (empty($options)) {
            $archivepath = "secrets.tar.gz";
        } elseif (count($options) == 1) {
            $archivepath = $options[0];
        }
        $archivePath = $workspace->locate($archivepath);
        $manager->packer->unpack($archivePath);
    },
];

if (array_key_exists($command, $handlers)) {
    try {
        $handlers[$command]($options);
    } catch (\Exception $e) {
        echo "Error: " . $e->getMessage() . "\n";
        exit(1);
    }
} else {
    echo "Usage: secretm.sh <command> [options]\n\n";
    echo "Available commands:\n";
    echo "  add <path>       : Add a secret file or directory to the manager\n";
    echo "  import [file]    : Import secrets from secrets.tar.gz archive\n";
    echo "  export [dir]     : Export all secrets into secrets.tar.gz archive\n";
}
