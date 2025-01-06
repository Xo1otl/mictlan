<?php

namespace koemade\dbadapter;

use koemade\actor;
use koemade\util;

class ProfileService implements actor\ProfileService
{
    private \PDO $conn;
    private util\Logger $logger;

    public function __construct()
    {
        $this->conn = DBConnection::getInstance();
        $this->logger = util\Logger::getInstance();
    }

    public function save(string $actor_id, actor\ProfileInput $input)
    {
        // Check if the profile exists
        $profileExists = $this->profileExists($actor_id);

        // Start a database transaction
        $this->conn->beginTransaction();

        try {
            // Prepare data arrays
            $actorProfileData = [
                'display_name' => $input->displayName,
                'self_promotion' => $input->selfPromotion,
                'price' => $input->price,
                'status' => $input->status,
            ];

            $nsfwOptionsData = [
                'allowed' => $input->nsfwAllowed,
                'price' => $input->nsfwPrice,
                'extreme_allowed' => $input->extremeAllowed,
                'extreme_surcharge' => $input->extremeSurcharge,
            ];

            $profileImageData = [
                'mime_type' => $input->profileImageMimeType,
                'size' => $input->profileImageSize,
                'path' => $input->profileImagePath,
            ];

            // Update or insert actor_profiles
            if ($profileExists) {
                $this->updateActorProfile($actor_id, $actorProfileData);
            } else {
                $this->insertActorProfile($actor_id, $actorProfileData);
            }

            // Update or insert nsfw_options
            if ($profileExists) {
                $this->updateNsfwOptions($actor_id, $nsfwOptionsData);
            } else {
                $this->insertNsfwOptions($actor_id, $nsfwOptionsData);
            }

            // Update or insert profile_images
            if ($profileExists) {
                $this->updateProfileImage($actor_id, $profileImageData);
            } else {
                $this->insertProfileImage($actor_id, $profileImageData);
            }

            // Commit the transaction
            $this->conn->commit();
            $this->logger->info("Profile saved for actor_id: $actor_id");
        } catch (\Exception $e) {
            // Rollback the transaction on error
            $this->conn->rollBack();
            $this->logger->error("Failed to save profile for actor_id: $actor_id. Error: " . $e->getMessage());
            throw $e; // Rethrow the exception after logging
        }
    }

    private function profileExists(string $actor_id): bool
    {
        $stmt = $this->conn->prepare("SELECT 1 FROM actor_profiles WHERE account_id = :actor_id LIMIT 1");
        $stmt->bindParam(':actor_id', $actor_id, \PDO::PARAM_INT);
        $stmt->execute();
        return $stmt->fetch() !== false;
    }

    private function updateActorProfile(string $actor_id, array $data)
    {
        // Build the SET clause with only non-null values
        $setClauses = [];
        $params = [];
        foreach ($data as $key => $value) {
            if ($value !== null) {
                $setClauses[] = "$key = :$key";
                $params[":$key"] = $value;
            }
        }

        if (!empty($setClauses)) {
            $sql = "UPDATE actor_profiles SET " . implode(', ', $setClauses) . " WHERE account_id = :actor_id";
            $params[":actor_id"] = $actor_id;
            $stmt = $this->conn->prepare($sql);
            $stmt->execute($params);
            $this->logger->debug("Updated actor_profiles for actor_id: $actor_id");
        }
    }

    private function insertActorProfile(string $actor_id, array $data)
    {
        $this->logger->debug($data);

        // Add account_id to the insert data
        $data['account_id'] = $actor_id;

        // Remove null values
        $data = array_filter($data, function ($value) {
            return $value !== null;
        });

        // Prepare columns and placeholders
        $columns = implode(', ', array_keys($data));
        $placeholders = ':' . implode(', :', array_keys($data));

        // Insert statement
        $sql = "INSERT INTO actor_profiles ($columns) VALUES ($placeholders)";
        $stmt = $this->conn->prepare($sql);
        $this->logger->debug($stmt);
        $stmt->execute($data);
        $this->logger->debug("Inserted new actor_profiles record for actor_id: $actor_id");
    }

    private function updateNsfwOptions(string $actor_id, array $data)
    {
        // Build the SET clause with only non-null values
        $setClauses = [];
        $params = [];
        foreach ($data as $key => $value) {
            if ($value !== null) {
                $setClauses[] = "$key = :$key";
                $params[":$key"] = $value;
            }
        }

        if (!empty($setClauses)) {
            $sql = "UPDATE nsfw_options SET " . implode(', ', $setClauses) . " WHERE account_id = :actor_id";
            $params[":actor_id"] = $actor_id;
            $stmt = $this->conn->prepare($sql);
            $stmt->execute($params);
            $this->logger->debug("Updated nsfw_options for actor_id: $actor_id");
        }
    }

    private function insertNsfwOptions(string $actor_id, array $data)
    {
        // Remove null values
        $data = array_filter($data, function ($value) {
            return $value !== null;
        });

        // Add account_id to the insert data
        $data['account_id'] = $actor_id;

        // Prepare columns and placeholders
        $columns = implode(', ', array_keys($data));
        $placeholders = ':' . implode(', :', array_keys($data));

        // Insert statement
        $sql = "INSERT INTO nsfw_options ($columns) VALUES ($placeholders)";
        $stmt = $this->conn->prepare($sql);
        $stmt->execute($data);
        $this->logger->debug("Inserted new nsfw_options record for actor_id: $actor_id");
    }

    private function updateProfileImage(string $actor_id, array $data)
    {
        // Build the SET clause with only non-null values
        $setClauses = [];
        $params = [];
        foreach ($data as $key => $value) {
            if ($value !== null) {
                $setClauses[] = "$key = :$key";
                $params[":$key"] = $value;
            }
        }

        if (!empty($setClauses)) {
            $sql = "UPDATE profile_images SET " . implode(', ', $setClauses) . " WHERE account_id = :actor_id";
            $params[":actor_id"] = $actor_id;
            $stmt = $this->conn->prepare($sql);
            $stmt->execute($params);
            $this->logger->debug("Updated profile_images for actor_id: $actor_id");
        }
    }

    private function insertProfileImage(string $actor_id, array $data)
    {
        // Remove null values
        $data = array_filter($data, function ($value) {
            return $value !== null;
        });

        // Add account_id to the insert data
        $data['account_id'] = $actor_id;

        // Prepare columns and placeholders
        $columns = implode(', ', array_keys($data));
        $placeholders = ':' . implode(', :', array_keys($data));

        // Insert statement
        $sql = "INSERT INTO profile_images ($columns) VALUES ($placeholders)";
        $stmt = $this->conn->prepare($sql);
        $stmt->execute($data);
        $this->logger->debug("Inserted new profile_images record for actor_id: $actor_id");
    }
}
