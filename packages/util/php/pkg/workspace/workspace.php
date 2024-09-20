<?php

namespace util\workspace;

interface Workspace
{
    function root(): string;
    function locate(string $path, $allowNonExistent = false): string;
}
