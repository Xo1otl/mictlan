<?php
namespace util\logger;

const ANSI_RESET = "\033[0m";
const ANSI_GREEN = "\033[32m";
const ANSI_BLUE = "\033[34m";
const ANSI_YELLOW = "\033[33m";
const ANSI_RED = "\033[31m";
const ANSI_MAGENTA = "\033[35m";
const ANSI_WHITE_ON_RED = "\033[97;41m";

$dateFormat = "D M  j H:i:s Y";
$output = "[%datetime%] %message%" . ANSI_RESET . "\n";

$formatter = new \Monolog\Formatter\LineFormatter($output, $dateFormat);
$handler = new \Monolog\Handler\StreamHandler('php://stdout', \Monolog\Level::Debug);
$handler->setFormatter($formatter);

global $log;
$log = new \Monolog\Logger("logger");
$log->pushHandler($handler);

function getCallerInfo()
{
    $backtrace = debug_backtrace(DEBUG_BACKTRACE_IGNORE_ARGS, 2);
    if (isset($backtrace[1])) {
        $caller = $backtrace[1];
        return " in " . ($caller['file'] ?? 'unknown') . " on line " . ($caller['line'] ?? 'unknown');
    }
    return "error in getCallerInfo";
}

class Info
{
    public function __construct(...$msgs)
    {
        global $log;
        $log->Info(ANSI_GREEN . "Info: " . implode(' ', array_map('json_encode', $msgs)) . getCallerInfo());
    }
}

class Debug
{
    public function __construct(...$msgs)
    {
        global $log;
        $log->Debug(ANSI_BLUE . "Debug: " . implode(' ', array_map('json_encode', $msgs)) . getCallerInfo());
    }
}

class Imp
{
    public function __construct(...$msgs)
    {
        global $log;
        $log->Info(ANSI_MAGENTA . "Important: " . implode(' ', array_map('json_encode', $msgs)) . getCallerInfo());
    }
}

class Warn
{
    public function __construct(...$msgs)
    {
        global $log;
        $log->Warning(ANSI_YELLOW . "Warning: " . implode(' ', array_map('json_encode', $msgs)) . getCallerInfo());
    }
}

class Err
{
    public function __construct(...$msgs)
    {
        global $log;
        $log->Error(ANSI_RED . "Error: " . implode(' ', array_map('json_encode', $msgs)) . getCallerInfo());
    }
}

class Fatal
{
    public function __construct(...$msgs)
    {
        global $log;
        $log->Critical(ANSI_WHITE_ON_RED . "Fatal: " . implode(' ', array_map('json_encode', $msgs)) . getCallerInfo());
    }
}