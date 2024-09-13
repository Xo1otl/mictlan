<?php

namespace voice;

interface Repo
{
    function upload(Input $input);

    /**
     * @param \common\Id $accountId
     * @return Voice[]
     */
    function getAll(\common\Id $accountId): array;

    function editTag(EditTagInput $input);

    function deleteById(\common\Id $voiceId, \common\Id $accountId);
}
