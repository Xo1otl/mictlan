<?php

namespace secretm;

use util\workspace;

class Cli implements VCS, PathRepo, Packer
{
    private $secretsRegFile;
    private $rootGitignore;
    private $workspaceFolder;

    public function __construct(workspace\Workspace $workspace)
    {
        $this->workspaceFolder = $workspace->root();
        $this->secretsRegFile = $this->workspaceFolder . DIRECTORY_SEPARATOR . "build" . DIRECTORY_SEPARATOR . "secrets.json";
        $this->rootGitignore = $this->workspaceFolder . DIRECTORY_SEPARATOR . ".gitignore";
    }

    public function ignoreIfNot(string $filepath)
    {
        $currentDir = getcwd();
        chdir($this->workspaceFolder);

        // Use git check-ignore to see if the file is ignored
        $output = [];
        $return_var = 0;

        exec('git check-ignore ' . escapeshellarg($filepath), $output, $return_var);

        if ($return_var !== 0) {
            // The file is not ignored, so add it to .gitignore
            file_put_contents($this->rootGitignore, "\n" . $filepath . PHP_EOL, FILE_APPEND | LOCK_EX);
            echo "Added $filepath to .gitignore\n";
        } else {
            echo "$filepath is already ignored.\n";
        }

        chdir($currentDir);
    }

    public function register(string $path)
    {
        // Ensure build directory exists
        $buildDir = dirname($this->secretsRegFile);
        if (!is_dir($buildDir)) {
            mkdir($buildDir, 0777, true);
        }

        // Check if secrets.json exists, if not, create it with an empty array
        if (!file_exists($this->secretsRegFile)) {
            file_put_contents($this->secretsRegFile, json_encode([]));
        }

        // Read the contents of secrets.json
        $secrets = json_decode(file_get_contents($this->secretsRegFile), true);

        if (!in_array($path, $secrets)) {
            $secrets[] = $path;
            file_put_contents($this->secretsRegFile, json_encode($secrets, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES));
            echo "Registered $path\n";
        } else {
            echo "$path is already registered\n";
        }
    }

    public function listAll(): array
    {
        if (!file_exists($this->secretsRegFile)) {
            echo "No secrets registered yet.\n";
            return [];
        }

        $secrets = json_decode(file_get_contents($this->secretsRegFile), true);

        if (!is_array($secrets)) {
            echo "Invalid secrets.json format.\n";
            return [];
        }

        return $secrets;
    }

    // compress and decompress are executed in workspaceFolder to properly preserve path information
    public function pack(array $filepaths, string $archivePath)
    {
        $currentDir = getcwd();
        chdir($this->workspaceFolder);

        if (empty($filepaths)) {
            echo "No files to compress.\n";
            chdir($currentDir);
            return;
        }

        // Build the tar command
        $files = implode(' ', array_map('escapeshellarg', $filepaths));
        $archivePath = escapeshellarg($archivePath);

        $cmd = "tar -czf $archivePath $files";

        exec($cmd, $output, $return_var);

        if ($return_var === 0) {
            echo "Compressed files into $archivePath\n";
        } else {
            echo "Error compressing files. Command: $cmd\n";
        }

        chdir($currentDir);
    }

    public function unpack(string $archivePath)
    {
        $currentDir = getcwd();
        chdir($this->workspaceFolder);

        if (!file_exists($archivePath)) {
            echo "Archive file $archivePath does not exist.\n";
            chdir($currentDir);
            return;
        }

        // アーカイブ内のファイルリストを取得
        $archiveFiles = [];
        $cmdList = "tar -tzf " . escapeshellarg($archivePath);
        exec($cmdList, $archiveFiles, $return_var);

        if ($return_var !== 0) {
            echo "Error listing archive contents. Command: $cmdList\n";
            chdir($currentDir);
            return;
        }

        // 登録されたシークレットのパスを取得
        $registeredSecrets = $this->listAll();

        // パスを正規化（相対パスや絶対パスの違いを吸収）
        $normalizedArchiveFiles = array_map(function ($path) {
            return ltrim($path, './'); // './'を削除
        }, $archiveFiles);

        $normalizedRegisteredSecrets = array_map(function ($path) {
            return ltrim($path, './'); // './'を削除
        }, $registeredSecrets);

        // リストをソートして比較
        sort($normalizedArchiveFiles);
        sort($normalizedRegisteredSecrets);

        if ($normalizedArchiveFiles !== $normalizedRegisteredSecrets) {
            echo "The contents of the archive do not match the registered secrets.\n";
            chdir($currentDir);
            return;
        }

        // アーカイブを解凍
        $archivePathEscaped = escapeshellarg($archivePath);
        $cmdExtract = "tar -xzf $archivePathEscaped";

        exec($cmdExtract, $output, $return_var);

        if ($return_var === 0) {
            echo "Decompressed archive $archivePath\n";
        } else {
            echo "Error decompressing archive. Command: $cmdExtract\n";
        }

        chdir($currentDir);
    }
}
