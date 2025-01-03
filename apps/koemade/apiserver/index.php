<?php
use Psr\Http\Message\ResponseInterface as Response;
use Psr\Http\Message\ServerRequestInterface as Request;
use Slim\Factory\AppFactory;
use koemade\auth;
use koemade\middleware;

require __DIR__ . '/../kernel/bootstrap.php';

$app = AppFactory::create();

$tokenService = new auth\JWTService($secretKey);

$app->add(new middleware\SlimAuth($tokenService));

// TODO: 認証認可はここで行う、ミドルウェアがclaimsをリクエストに追加しているのでroleで判定してドメインロジック呼び出したり呼び出さなかったり、引数にsubなどの情報を渡したり
$app->get('/', function (Request $request, Response $response, $args) {
    $response->getBody()->write("Hello world!");
    return $response;
});

$app->run();
