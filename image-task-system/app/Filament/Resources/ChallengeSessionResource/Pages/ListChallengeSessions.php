<?php

namespace App\Filament\Resources\ChallengeSessionResource\Pages;

use App\Filament\Resources\ChallengeSessionResource;
use Filament\Actions;
use Filament\Resources\Pages\ListRecords;

class ListChallengeSessions extends ListRecords
{
    protected static string $resource = ChallengeSessionResource::class;

    protected function getHeaderActions(): array
    {
        return [
            Actions\CreateAction::make(),
        ];
    }
}
