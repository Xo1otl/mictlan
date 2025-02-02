<?php
use Psr\Http\Message\ResponseInterface as Response;
use Psr\Http\Message\ServerRequestInterface as Request;
use Slim\Factory\AppFactory;
use koemade\auth;
use koemade\middleware;
use koemade\dbadapter\SearchService;
use koemade\dbadapter\QueryRepository;
use koemade\query\search\ActorsParams;
use koemade\query\search\VoicesParams;
use koemade\query\valueObjects\Tag;

require __DIR__ . '/../koemade/kernel/bootstrap.php';

$app = AppFactory::create();

$tokenService = new auth\JWTService($secretKey);

$app->add(new middleware\SlimAuth($tokenService));
$app->add(new middleware\SlimCORS());

// サービスを初期化
$searchService = new SearchService();
$queryRepository = new QueryRepository();

// /search-voices route
$app->get('/search-voices', function (Request $request, Response $response, array $args) use ($searchService) {
    $queryParams = $request->getQueryParams();

    // Parse title
    $title = $queryParams['title'] ?? null;

    // Parse tags
    $tags = [];
    // クエリパラメータのtagsが配列である場合
    // パラメータの書式例：tags[category1]=name1&tags[category2]=name2
    if (isset($queryParams['tags']) && is_array($queryParams['tags'])) {
        foreach ($queryParams['tags'] as $category => $name) {
            $category = htmlspecialchars($category, ENT_QUOTES, 'UTF-8');
            $name = htmlspecialchars($name, ENT_QUOTES, 'UTF-8');
            if (empty($category)) {
                $response = $response->withHeader('Content-Type', 'application/json');
                $response->getBody()->write(json_encode(['error' => 'Invalid tag format']));
                return $response->withStatus(400);
            }
            $tags[] = new Tag($category, $name);
        }
    }

    // Parse page
    $page = isset($queryParams['page']) && is_numeric($queryParams['page']) ? (int) $queryParams['page'] : 1;

    // Create VoicesParams
    $voicesParams = new VoicesParams($title, $tags, $page);

    try {
        $voices = $searchService->voices($voicesParams);
        $responseData = [
            'total_matches' => count($voices), // 総一致件数を設定
            'items' => $voices // アイテムを設定
        ];
        $response = $response->withHeader('Content-Type', 'application/json');
        $response->getBody()->write(json_encode($responseData, JSON_UNESCAPED_UNICODE));
        return $response;
    } catch (\Exception $e) {
        $response = $response->withHeader('Content-Type', 'application/json');
        $response->getBody()->write(json_encode(['error' => 'Internal Server Error']));
        return $response->withStatus(500);
    }
});

// /search-actors route
$app->get('/search-actors', function (Request $request, Response $response, array $args) use ($searchService) {
    $queryParams = $request->getQueryParams();

    // Parse name_like
    $nameLike = $queryParams['name_like'] ?? null;

    // Parse status
    $status = $queryParams['status'] ?? null;

    // Parse nsfw_options
    $nsfwOptions = [];
    if (isset($queryParams['nsfw_allowed']) && is_string($queryParams['nsfw_allowed'])) {
        $nsfwOptions['allowed'] = filter_var($queryParams['nsfw_allowed'], FILTER_VALIDATE_BOOLEAN);
    }
    if (isset($queryParams['nsfw_extreme_allowed']) && is_string($queryParams['nsfw_extreme_allowed'])) {
        $nsfwOptions['extreme_allowed'] = filter_var($queryParams['nsfw_extreme_allowed'], FILTER_VALIDATE_BOOLEAN);
    }

    // Parse page
    $page = isset($queryParams['page']) && is_numeric($queryParams['page']) ? (int) $queryParams['page'] : 1;

    // Create ActorsParams
    $actorsParams = new ActorsParams($nameLike, $status, $page, $nsfwOptions);

    try {
        $actors = $searchService->actors($actorsParams);
        $responseData = [
            'total_matches' => count($actors), // 総一致件数を設定
            'items' => $actors // アイテムを設定
        ];
        $response = $response->withHeader('Content-Type', 'application/json');
        $response->getBody()->write(json_encode($responseData, JSON_UNESCAPED_UNICODE));
        return $response;
    } catch (\Exception $e) {
        $response = $response->withHeader('Content-Type', 'application/json');
        $response->getBody()->write(json_encode(['error' => 'Internal Server Error']));
        return $response->withStatus(500);
    }
});

// /voice/{id} route
$app->get('/voice/{id}', function (Request $request, Response $response, array $args) use ($queryRepository) {
    $id = $args['id'];
    if (!is_numeric($id)) {
        $response = $response->withHeader('Content-Type', 'application/json');
        $response->getBody()->write(json_encode(['error' => 'Invalid ID']));
        return $response->withStatus(400);
    }

    try {
        $voice = $queryRepository->findVoiceWithTagsByID((int) $id);
        if ($voice === null) {
            $response = $response->withHeader('Content-Type', 'application/json');
            $response->getBody()->write(json_encode(['error' => 'Voice not found']));
            return $response->withStatus(404);
        }
        $response = $response->withHeader('Content-Type', 'application/json');
        $response->getBody()->write(json_encode($voice, JSON_UNESCAPED_UNICODE));
        return $response;
    } catch (\Exception $e) {
        $response = $response->withHeader('Content-Type', 'application/json');
        $response->getBody()->write(json_encode(['error' => 'Internal Server Error']));
        return $response->withStatus(500);
    }
});

// /actor/{id} route
$app->get('/actor/{id}', function (Request $request, Response $response, array $args) use ($queryRepository) {
    $id = $args['id'];
    if (!is_numeric($id)) {
        $response = $response->withHeader('Content-Type', 'application/json');
        $response->getBody()->write(json_encode(['error' => 'Invalid ID']));
        return $response->withStatus(400);
    }

    try {
        $actorFeed = $queryRepository->actorFeed((int) $id);
        if ($actorFeed === null) {
            $response = $response->withHeader('Content-Type', 'application/json');
            $response->getBody()->write(json_encode(['error' => 'Actor not found']));
            return $response->withStatus(404);
        }
        $responseData = [
            'actor' => $actorFeed->actor,
            'sample_voices' => [ // スネークケースに統一
                'total_matches' => count($actorFeed->sampleVoices ?? []), // 総一致件数を設定
                'items' => $actorFeed->sampleVoices ?? [] // アイテムを設定
            ]
        ];
        $response = $response->withHeader('Content-Type', 'application/json');
        $response->getBody()->write(json_encode($responseData, JSON_UNESCAPED_UNICODE));
        return $response;
    } catch (\Exception $e) {
        $response = $response->withHeader('Content-Type', 'application/json');
        $response->getBody()->write(json_encode(['error' => 'Internal Server Error']));
        return $response->withStatus(500);
    }
});

$app->run();
