from django.contrib import admin
from article.models import Article, Comment


class CommentAdmin(admin.ModelAdmin):
    list_display = ['article', 'content']

    class Meta:
        model = Comment


admin.site.register(Article)
admin.site.register(Comment, CommentAdmin)
