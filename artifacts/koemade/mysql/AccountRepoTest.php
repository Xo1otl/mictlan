<?php

namespace mysql;

require __DIR__ . '/../vendor/autoload.php';

function TestAccountRepo()
{
    $db = new AccountRepo();
    $username = new \auth\Username(\utils\generateUUID() . "@gmail.com");
    try {
        $plainText = \utils\generatePlainPassword();
        \logger\imp($plainText);
    } catch (\Exception $e) {
        \logger\fatal($e);
    }
    $password = new \auth\Password($plainText);
    $input = new \auth\CredentialInput($username, $password);
    try {
        $db->add($input);
    } catch (\Exception $e) {
        \logger\fatal($e);
    }
}

TestAccountRepo();
