<?php

namespace logger;

function initLogger(): \Monolog\Logger
{
    $levels = [
        \Monolog\Logger::DEBUG => "\033[1;34m%s\033[0m", // 青
        \Monolog\Logger::INFO => "\033[1;32m%s\033[0m", // 緑
        \Monolog\Logger::NOTICE => "\033[1;35m%s\033[0m", // マゼンタ
        \Monolog\Logger::WARNING => "\033[1;33m%s\033[0m", // 黄色
        \Monolog\Logger::ERROR => "\033[1;31m%s\033[0m", // 赤
        \Monolog\Logger::CRITICAL => "\033[1;37;41m%s\033[0m", // 白（赤背景）
        \Monolog\Logger::ALERT => "\033[1;37;41m%s\033[0m", // 白（赤背景）
        \Monolog\Logger::EMERGENCY => "\033[1;37;41m%s\033[0m" // 白（赤背景）
    ];

    $logger = new \Monolog\Logger('logger');
    $handler = new \Monolog\Handler\StreamHandler('php://stdout', \Monolog\Logger::DEBUG);

    $handler->setFormatter(new class($levels) extends \Monolog\Formatter\LineFormatter {
        private array $levels;

        public function __construct($levels)
        {
            parent::__construct("[%datetime%] %message%\n", "D M  j H:i:s Y", true, true);
            $this->levels = $levels;
        }

        public function format(array $record): string
        {
            $output = parent::format($record);
            $level = $record['level'];
            $levelColor = $this->levels[$level] ?? '%s';
            return sprintf($levelColor, $output);
        }
    });

    $logger->pushHandler($handler);
    return $logger;
}


function getCallerInfo(): array
{
    $debugBacktrace = debug_backtrace(DEBUG_BACKTRACE_IGNORE_ARGS, 3);
    if (isset($debugBacktrace[2])) {
        $caller = $debugBacktrace[2];
        $file = basename($caller['file']);
        $line = $caller['line'];
    } else {
        $file = '???';
        $line = 0;
    }
    return [$file, $line];
}

function formatMessage($level, $message): string
{
    list($file, $line) = getCallerInfo();
    return sprintf("%s:%d | %s: %s", $file, $line, strtoupper($level), $message);
}

function info(...$args)
{
    $logger = initLogger();
    $message = formatMessage('info', implode(' ', array_map('json_encode', $args)));
    $logger->info($message);
}

function debug(...$args)
{
    $logger = initLogger();
    $message = formatMessage('debug', implode(' ', array_map('json_encode', $args)));
    $logger->debug($message);
}

function imp(...$args)
{
    $logger = initLogger();
    $message = formatMessage('important', implode(' ', array_map('json_encode', $args)));
    $logger->notice($message);
}

function warn(...$args)
{
    $logger = initLogger();
    $message = formatMessage('warn', implode(' ', array_map('json_encode', $args)));
    $logger->warning($message);
}

function err(...$args)
{
    $logger = initLogger();
    $message = formatMessage('error', implode(' ', array_map('json_encode', $args)));
    $logger->error($message, ['stack' => debug_backtrace(DEBUG_BACKTRACE_IGNORE_ARGS)]);
}

function fatal(...$args)
{
    $logger = initLogger();
    $message = formatMessage('debug', implode(' ', array_map('json_encode', $args)));
    $logger->critical($message);
    throw new \RuntimeException($message);
}
