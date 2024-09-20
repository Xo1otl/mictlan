<?php

require __DIR__ . "/../../vendor/autoload.php";

use util\logger;

new logger\Debug("Hello, world!");
new logger\Info("ExampleObject", ["key" => "value"]);
