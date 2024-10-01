<?php

namespace util\workspace;

class PHP implements Workspace
{
    public $workFile = 'workspace.php';
    public $root = "";
    public $use = [];
    function __construct()
    {
        $path = getcwd();
        while ($path !== '/' && $path !== '') {
            if (file_exists($path . DIRECTORY_SEPARATOR . $this->workFile)) {
                $this->root = $path;
                require $path . DIRECTORY_SEPARATOR . $this->workFile;
                $this->use = $use;
                return;
            }
            $path = dirname($path);
        }
        throw new \Exception("Workspace root (containing php.work) not found.");
    }

    function root(): string
    {
        return $this->root;
    }

    function locate(string $path, $allowNonExistent = false): string
    {
        // 絶対パスの場合
        if (strpos($path, '/') === 0 || (strlen($path) > 1 && $path[1] === ':')) {
            $absolutePath = $allowNonExistent ? $path : realpath($path);
            if ($absolutePath === false) {
                throw new \Exception("The specified absolute path does not exist or is not accessible: $path");
            }
            if (strpos($absolutePath, $this->root) === 0) {
                return substr($absolutePath, strlen($this->root) + 1);
            }
            throw new \Exception("The specified path is outside the workspace: $path");
        }

        // 相対パスの場合
        $absolutePath = $allowNonExistent ? $this->root . DIRECTORY_SEPARATOR . $path : realpath($this->root . DIRECTORY_SEPARATOR . $path);
        if ($absolutePath === false) {
            throw new \Exception("The specified relative path does not exist or is not accessible: $path");
        }
        if (strpos($absolutePath, $this->root) === 0) {
            return substr($absolutePath, strlen($this->root) + 1);
        }
        throw new \Exception("The specified path is outside the workspace: $path");
    }

    function packages(): array
    {
        return $this->use;
    }
}
