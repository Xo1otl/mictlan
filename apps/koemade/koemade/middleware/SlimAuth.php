<?php

namespace koemade\middleware;

use Psr\Http\Message\ResponseInterface as Response;
use Psr\Http\Message\ServerRequestInterface as Request;
use Psr\Http\Server\RequestHandlerInterface as RequestHandler;
use Slim\Psr7\Response as SlimResponse;
use koemade\auth;

class SlimAuth
{
    private $tokenService;

    public function __construct(auth\TokenService $tokenService)
    {
        $this->tokenService = $tokenService;
    }

    public function __invoke(Request $request, RequestHandler $handler): Response
    {
        $authHeader = $request->getHeaderLine('Authorization');

        // トークンが提供されていない場合、そのままリクエストを処理
        if (empty($authHeader) || !preg_match('/Bearer\s(\S+)/', $authHeader, $matches)) {
            return $handler->handle($request);
        }

        $token = $matches[1];

        try {
            // トークンを検証
            $claims = $this->tokenService->verify($token);
            $request = $request->withAttribute('claims', $claims);
        } catch (\Exception $e) {
            // 不正なトークンの場合、エラーレスポンスを返す
            return $this->createUnauthorizedResponse($e->getMessage());
        }

        return $handler->handle($request);
    }

    private function createUnauthorizedResponse(string $message): Response
    {
        $response = new SlimResponse();
        $response->getBody()->write(json_encode(['error' => $message]));
        return $response
            ->withStatus(401)
            ->withHeader('Content-Type', 'application/json');
    }
}
