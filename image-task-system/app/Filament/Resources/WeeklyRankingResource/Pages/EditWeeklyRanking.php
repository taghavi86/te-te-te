<?php

namespace App\Filament\Resources\WeeklyRankingResource\Pages;

use App\Filament\Resources\WeeklyRankingResource;
use Filament\Actions;
use Filament\Resources\Pages\EditRecord;

class EditWeeklyRanking extends EditRecord
{
    protected static string $resource = WeeklyRankingResource::class;

    protected function getHeaderActions(): array
    {
        return [
            Actions\DeleteAction::make(),
        ];
    }
}
