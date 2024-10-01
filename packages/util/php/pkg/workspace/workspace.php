<?php

namespace util\workspace;

interface Workspace
{
    function root(): string;
    function packages(): array;
    function locate(string $path, $allowNonExistent = false): string;
}
