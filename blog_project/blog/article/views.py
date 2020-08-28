from django.shortcuts import render
from article.models import Article, Comment


# Create your views here.


def article(request):
    articles = {article: Comment.objects.filter(
        article=article) for article in Article.objects.all()}
    context = {'articles': articles}
    return render(request, 'article/article.html', context)
