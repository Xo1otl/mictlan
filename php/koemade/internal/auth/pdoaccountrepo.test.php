<?php

require __DIR__ . '/../../vendor/autoload.php';

$accountRepo = new \auth\PdoAccountRepo();
$username = new \auth\Username("abcd@gmail.com");
$password = new \auth\Password("Abcd1234*");
$signUpInput = new \auth\SignUpInput($username, $password);
$id = $accountRepo->add($signUpInput);
new \logger\Debug($id);

$newPassword = new \auth\Password("Abcd12345*");
$editPasswordInput = new \auth\EditPasswordInput($username, $password, $newPassword);
$accountRepo->editPassword($editPasswordInput);

$account = $accountRepo->findByUsername($username);
new \logger\Debug("verified? " . ($account->password->verify("Abcd12345*") ? "true" : "false"));

$username1 = new \auth\Username("abcd2@gmail.com");
$password1 = new \auth\Password("Abcd1234*");
$signUpInput1 = new \auth\SignUpInput($username1, $password1);
$id1 = $accountRepo->add($signUpInput1);
new \logger\Debug($id1);

$accounts = $accountRepo->allAccounts();
new \logger\Debug(json_encode($accounts));

$accountRepo->deleteByUsername($username);
$accountRepo->deleteById($id1);