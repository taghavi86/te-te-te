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
        Schema::create('badges', function (Blueprint $table) {
            $table->id();
            $table->string('name'); // نام نشان
            $table->text('description')->nullable(); // توضیحات
            $table->string('icon_path')->nullable(); // مسیر آیکون
            $table->string('badge_type')->default('achievement'); // نوع: achievement, milestone
            $table->integer('requirement_value')->default(0); // مقدار مورد نیاز برای دریافت
            $table->string('requirement_type')->nullable(); // نوع شرط: tasks_completed, images_processed, etc.
            $table->boolean('is_active')->default(true);
            $table->timestamps();
        });
        
        // جدول واسط برای رابطه چند به چند کاربران و نشان‌ها
        Schema::create('user_badges', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained()->onDelete('cascade');
            $table->foreignId('badge_id')->constrained()->onDelete('cascade');
            $table->timestamp('earned_at')->useCurrent();
            $table->timestamps();
            
            $table->unique(['user_id', 'badge_id']);
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('user_badges');
        Schema::dropIfExists('badges');
    }
};
