<?php

namespace App\Filament\Resources\UserLevelResource\Pages;

use App\Filament\Resources\UserLevelResource;
use Filament\Actions;
use Filament\Resources\Pages\EditRecord;

class EditUserLevel extends EditRecord
{
    protected static string $resource = UserLevelResource::class;

    protected function getHeaderActions(): array
    {
        return [
            Actions\DeleteAction::make(),
        ];
    }
}
