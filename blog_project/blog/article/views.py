from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.db.models.query_utils import Q
from article.models import Article, Comment
from article.forms import ArticleForm


# Create your views here.


def article(request):
    articles = {article: Comment.objects.filter(
        article=article) for article in Article.objects.all()}
    context = {'articles': articles}
    return render(request, 'article/article.html', context)


def articleCreate(request):
    template = 'article/articleUpdate.html'
    if request.method == 'GET':
        return render(request, template, {'articleForm': ArticleForm()})
    articleForm = ArticleForm(request.POST)
    if not articleForm.is_valid():
        return render(request, template, {'articleForm': articleForm})
    articleForm.save()
    messages.success(request, '文章已新增')
    return redirect('article:article')


def articleRead(request, articleId):
    article = get_object_or_404(Article, id=articleId)
    context = {
        'article': article,
        'comments': Comment.objects.filter(article=article)
    }
    return render(request, 'article/articleRead.html', context)


def articleUpdate(request, articleId):
    article = get_object_or_404(Article, id=articleId)
    template = 'article/articleUpdate.html'
    if request.method == 'GET':
        articleForm = ArticleForm(instance=article)
        return render(request, template, {'articleForm': articleForm})

    # POST
    articleForm = ArticleForm(request.POST, instance=article)
    if not articleForm.is_valid():
        return render(request, template, {'articleForm': articleForm})
    articleForm.save()
    messages.success(request, '文章已修改')
    return redirect('article:articleRead', articleId=articleId)


def articleDelete(request, articleId):
    if request.method == 'GET':
        return redirect('article:article')

    #  POST
    article = get_object_or_404(Article, id=articleId)
    article.delete()
    messages.success(request, '文章已刪除')
    return redirect('article:article')


def articleSearch(request):
    searchTerm = request.GET.get('searchTerm')
    articles = Article.objects.filter(
        Q(title__icontains=searchTerm) | Q(content__icontains=searchTerm))
    context = {'articles': articles, 'searchTerm': searchTerm}
    print(context)
    return render(request, 'article/articleSearch.html', context)
