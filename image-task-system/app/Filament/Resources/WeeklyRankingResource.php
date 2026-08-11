<?php

namespace App\Filament\Resources;

use App\Filament\Resources\WeeklyRankingResource\Pages;
use App\Filament\Resources\WeeklyRankingResource\RelationManagers;
use App\Models\WeeklyRanking;
use Filament\Forms;
use Filament\Forms\Form;
use Filament\Resources\Resource;
use Filament\Tables;
use Filament\Tables\Table;
use Illuminate\Database\Eloquent\Builder;
use Illuminate\Database\Eloquent\SoftDeletingScope;

class WeeklyRankingResource extends Resource
{
    protected static ?string $model = WeeklyRanking::class;

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
            'index' => Pages\ListWeeklyRankings::route('/'),
            'create' => Pages\CreateWeeklyRanking::route('/create'),
            'edit' => Pages\EditWeeklyRanking::route('/{record}/edit'),
        ];
    }
}
