<?php

namespace koemade\util;

use Monolog\Logger as MonologLogger;
use Monolog\Handler\StreamHandler;
use Monolog\Formatter\LineFormatter;

class Logger
{
    private static $instance;
    private $logger;

    private function __construct()
    {
        $dateFormat = "D M j H:i:s Y";
        $output = "[%datetime%] %message%" . self::ANSI_RESET . "\n";

        $formatter = new LineFormatter($output, $dateFormat);
        $handler = new StreamHandler('php://stdout', MonologLogger::DEBUG);
        $handler->setFormatter($formatter);

        $this->logger = new MonologLogger("logger");
        $this->logger->pushHandler($handler);
    }

    public static function getInstance()
    {
        if (!self::$instance) {
            self::$instance = new self();
        }
        return self::$instance;
    }

    public function info(...$msgs)
    {
        $this->log(MonologLogger::INFO, "Info: " . implode(' ', array_map('json_encode', $msgs)) . $this->getCallerInfo());
    }

    public function debug(...$msgs)
    {
        $this->log(MonologLogger::DEBUG, "Debug: " . implode(' ', array_map('json_encode', $msgs)) . $this->getCallerInfo());
    }

    public function warning(...$msgs)
    {
        $this->log(MonologLogger::WARNING, "Warning: " . implode(' ', array_map('json_encode', $msgs)) . $this->getCallerInfo());
    }

    public function error(...$msgs)
    {
        $this->log(MonologLogger::ERROR, "Error: " . implode(' ', array_map('json_encode', $msgs)) . $this->getCallerInfo());
    }

    public function critical(...$msgs)
    {
        $this->log(MonologLogger::CRITICAL, "Fatal: " . implode(' ', array_map('json_encode', $msgs)) . $this->getCallerInfo());
    }

    private function log($level, $message)
    {
        $this->logger->log($level, $message);
    }

    private function getCallerInfo()
    {
        $backtrace = debug_backtrace(DEBUG_BACKTRACE_IGNORE_ARGS, 2);
        if (isset($backtrace[1])) {
            $caller = $backtrace[1];
            return " in " . ($caller['file'] ?? 'unknown') . " on line " . ($caller['line'] ?? 'unknown');
        }
        return " error in getCallerInfo";
    }

    public const ANSI_RESET = "\033[0m";
    public const ANSI_GREEN = "\033[32m";
    public const ANSI_BLUE = "\033[34m";
    public const ANSI_YELLOW = "\033[33m";
    public const ANSI_RED = "\033[31m";
    public const ANSI_MAGENTA = "\033[35m";
    public const ANSI_WHITE_ON_RED = "\033[97;41m";
}
