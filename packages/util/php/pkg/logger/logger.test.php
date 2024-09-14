<?php

require __DIR__ . "/../../vendor/autoload.php";

new \logger\Debug("Hello, world!");
new \logger\Info("ExampleObject", ["key" => "value"]);
