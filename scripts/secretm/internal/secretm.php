<?php

class Manager
{
    public function __construct(public Git $git, public PathRepo $repo, public Tar $tar)
    {
    }

    public function add(string $path)
    {
        $this->git->ignoreIfNot($path);
        $this->repo->register($path);
    }

    public function pack(string $archivepath)
    {
        $files = $this->repo->listAll();
        $this->tar->compress($files, "$archivepath");
    }
}

interface Git
{
    public function ignoreIfNot(string $filepath);
}

interface PathRepo
{
    public function register(string $path);
    public function listAll(): array;
}

interface Tar
{
    public function compress(array $filepaths, string $archivepath);
    public function decompress(string $archivepath);
}
