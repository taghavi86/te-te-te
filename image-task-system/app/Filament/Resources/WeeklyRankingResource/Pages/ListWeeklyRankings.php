<?php

namespace App\Filament\Resources\WeeklyRankingResource\Pages;

use App\Filament\Resources\WeeklyRankingResource;
use Filament\Actions;
use Filament\Resources\Pages\ListRecords;

class ListWeeklyRankings extends ListRecords
{
    protected static string $resource = WeeklyRankingResource::class;

    protected function getHeaderActions(): array
    {
        return [
            Actions\CreateAction::make(),
        ];
    }
}
