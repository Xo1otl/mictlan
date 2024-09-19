<?php

class Manager
{
    public function __construct(public VCS $vcs, public PathRepo $repo, public Packer $packer)
    {
    }

    public function add(string $path)
    {
        $this->vcs->ignoreIfNot($path);
        $this->repo->register($path);
    }

    public function export(string $archivepath)
    {
        $files = $this->repo->listAll();
        $this->packer->pack($files, "$archivepath");
    }
}

interface VCS
{
    public function ignoreIfNot(string $filepath);
}

interface PathRepo
{
    public function register(string $path);
    public function listAll(): array;
}

interface Packer
{
    public function pack(array $filepaths, string $archivepath);
    public function unpack(string $archivepath);
}
