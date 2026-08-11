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
        Schema::create('invitations', function (Blueprint $table) {
            $table->id();
            $table->foreignId('inviter_id')->constrained('users')->onDelete('cascade'); // دعوت‌کننده
            $table->foreignId('invited_user_id')->nullable()->constrained('users')->onDelete('set null'); // کاربر دعوت شده
            $table->string('invite_code')->unique(); // کد دعوت
            $table->string('invite_link')->nullable(); // لینک دعوت
            $table->enum('status', ['pending', 'accepted', 'expired'])->default('pending');
            $table->boolean('reward_paid')->default(false); // آیا پاداش پرداخت شده
            $table->integer('reward_amount')->default(100000); // مبلغ پاداش به تومان
            $table->timestamp('expires_at')->nullable();
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('invitations');
    }
};
