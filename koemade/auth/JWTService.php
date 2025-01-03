<?php

namespace koemade\auth;

use Firebase\JWT\JWT;
use Firebase\JWT\Key;
use Firebase\JWT\ExpiredException;
use Firebase\JWT\SignatureInvalidException;

class JWTService implements TokenService
{
    private $publicKey; // 公開鍵（RS256の場合）または秘密鍵（HS256の場合）
    private $algorithm; // アルゴリズム（例: 'RS256', 'HS256'）

    public function __construct(string $publicKey, string $algorithm = 'HS256')
    {
        $this->publicKey = $publicKey;
        $this->algorithm = $algorithm;
    }

    /**
     * @inheritDoc
     */
    public function verify(string $token): Claims
    {
        try {
            $decoded = JWT::decode($token, new Key($this->publicKey, $this->algorithm));
            $claims = new Claims($decoded->sub, $decoded->exp, $decoded->iat, $decoded->nbf, $decoded->role, $decoded->status);
            return $claims;
        } catch (ExpiredException $e) {
            // トークンの有効期限が切れている場合
            throw new exceptions\ExpiredTokenException("Token has expired.");
        } catch (SignatureInvalidException $e) {
            // 署名が無効な場合
            throw new exceptions\InvalidTokenException("Invalid token signature.");
        } catch (\Exception $e) {
            // その他のエラー
            throw new exceptions\InvalidTokenException("Token verification failed: " . $e->getMessage());
        }
    }
}
