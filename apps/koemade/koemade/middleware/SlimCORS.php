<?php

namespace koemade\middleware;

use Psr\Http\Message\ResponseInterface as Response;
use Psr\Http\Message\ServerRequestInterface as Request;
use Slim\Psr7\Response as SlimResponse;

class SlimCORS
{
    public function __invoke(Request $request, $handler): Response
    {
        $origin = $request->getHeaderLine('Origin');

        // localhostとkoemade.netのすべてのサブドメインとポートを許可
        if (preg_match('/^(https?:\/\/)(localhost|.*\.koemade\.net)(:\d+)?$/', $origin)) {
            $response = $handler->handle($request);
            $response = $response->withHeader('Access-Control-Allow-Origin', $origin)
                ->withHeader('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type, Accept, Origin, Authorization')
                ->withHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, PATCH, OPTIONS')
                ->withHeader('Access-Control-Allow-Credentials', 'true');
            return $response;
        }

        // 許可されていないオリジンの場合は403 Forbiddenを返す
        $response = new SlimResponse();
        $response->getBody()->write(json_encode(['error' => 'Forbidden']));
        return $response->withStatus(403)->withHeader('Content-Type', 'application/json');
    }
}