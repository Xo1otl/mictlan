<?php
namespace logger;

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
    public function __construct(string|\Stringable $message)
    {
        global $log;
        $log->Info(ANSI_GREEN . "Info: " . $message . getCallerInfo());
    }
}

class Debug
{
    public function __construct(string|\Stringable $message)
    {
        global $log;
        $log->Debug(ANSI_BLUE . "Debug: " . $message . getCallerInfo());
    }
}

class Imp
{
    public function __construct(string|\Stringable $message)
    {
        global $log;
        $log->Info(ANSI_MAGENTA . "Important: " . $message . getCallerInfo());
    }
}

class Warn
{
    public function __construct(string|\Stringable $message)
    {
        global $log;
        $log->Warning(ANSI_YELLOW . "Warning: " . $message . getCallerInfo());
    }
}

class Err
{
    public function __construct(string|\Stringable $message)
    {
        global $log;
        $log->Error(ANSI_RED . "Error: " . $message . getCallerInfo());
    }
}

class Fatal
{
    public function __construct(string|\Stringable $message)
    {
        global $log;
        $log->Critical(ANSI_WHITE_ON_RED . "Fatal: " . $message . getCallerInfo());
    }
}