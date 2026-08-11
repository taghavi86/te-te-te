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
        Schema::create('tasks', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained()->onDelete('cascade');
            $table->foreignId('image_id')->constrained()->onDelete('cascade');
            $table->enum('status', ['pending', 'in_progress', 'submitted', 'approved', 'rejected'])->default('pending');
            $table->json('answers')->nullable(); // پاسخ‌های کاربر به صورت JSON
            $table->integer('total_score')->default(0); // امتیاز کل
            $table->integer('accuracy_score')->default(0); // امتیاز دقت
            $table->integer('speed_score')->default(0); // امتیاز سرعت
            $table->integer('tokens_earned')->default(0); // توکن‌های کسب شده
            $table->boolean('is_confirmed')->default(false); // آیا توسط مدیر تایید شده
            $table->foreignId('reviewed_by')->nullable()->constrained('users')->onDelete('set null');
            $table->text('review_notes')->nullable(); // یادداشت‌های بازبینی
            $table->timestamp('started_at')->nullable();
            $table->timestamp('completed_at')->nullable();
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('tasks');
    }
};
