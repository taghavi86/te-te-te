<?php

namespace App\Filament\Resources;

use App\Filament\Resources\ChallengeSessionResource\Pages;
use App\Filament\Resources\ChallengeSessionResource\RelationManagers;
use App\Models\ChallengeSession;
use Filament\Forms;
use Filament\Forms\Form;
use Filament\Resources\Resource;
use Filament\Tables;
use Filament\Tables\Table;
use Illuminate\Database\Eloquent\Builder;
use Illuminate\Database\Eloquent\SoftDeletingScope;

class ChallengeSessionResource extends Resource
{
    protected static ?string $model = ChallengeSession::class;

    protected static ?string $navigationIcon = 'heroicon-o-rectangle-stack';

    public static function form(Form $form): Form
    {
        return $form
            ->schema([
                //
            ]);
    }

    public static function table(Table $table): Table
    {
        return $table
            ->columns([
                //
            ])
            ->filters([
                //
            ])
            ->actions([
                Tables\Actions\EditAction::make(),
            ])
            ->bulkActions([
                Tables\Actions\BulkActionGroup::make([
                    Tables\Actions\DeleteBulkAction::make(),
                ]),
            ]);
    }

    public static function getRelations(): array
    {
        return [
            //
        ];
    }

    public static function getPages(): array
    {
        return [
            'index' => Pages\ListChallengeSessions::route('/'),
            'create' => Pages\CreateChallengeSession::route('/create'),
            'edit' => Pages\EditChallengeSession::route('/{record}/edit'),
        ];
    }
}
