<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('challenge_sessions', function (Blueprint $table) {
            $table->id();
            $table->string('name'); // نام چالش
            $table->text('description')->nullable();
            $table->timestamp('starts_at'); // زمان شروع
            $table->timestamp('ends_at'); // زمان پایان
            $table->decimal('bonus_multiplier', 3, 1)->default(2.0); // ضریب پاداش (مثلاً 2 برابر)
            $table->integer('max_bonus_per_user')->default(100); // حداکثر پاداش اضافی برای هر کاربر
            $table->enum('status', ['scheduled', 'active', 'completed'])->default('scheduled');
            $table->boolean('is_active')->default(true);
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('challenge_sessions');
    }
};
