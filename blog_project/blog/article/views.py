from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.db.models.query_utils import Q
from django.contrib.auth.decorators import login_required

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


@login_required
def articleLike(request, articleId):
    article = get_object_or_404(Article, id=articleId)
    if request.user not in article.likes.all():
        article.likes.add(request.user)
    return articleRead(request, articleId)


@login_required
def commentCreate(request, articleId):
    if request.method == 'GET':
        return articleRead(request, articleId)

    # POST
    comment = request.POST.get('comment')
    if comment:
        comment = comment.strip()
    if not comment:
        return redirect('article"articleRead', articleId=articleId)

    article = get_object_or_404(Article, id=articleId)
    Comment.objects.create(article=article, user=request.user, content=comment)
    return redirect('article:articleRead', articleId=articleId)


@login_required
def commentUpdate(request, commentId):
    commentToUpdate = get_object_or_404(Comment, id=commentId)
    article = get_object_or_404(Article, id=commentToUpdate.article.id)
    if request.method == 'GET':
        return articleRead(request, article.id)

    # POST
    if commentToUpdate.user != request.user:
        messages.error(request, '無修改權限')
        return redirect('article:articleRead', articleId=article.id)

    comment = request.POST.get('comment', ''.strip())
    if not comment:
        commentToUpdate.delete()
    else:
        commentToUpdate.content = comment
        commentToUpdate.save()
    return redirect('article:articleRead', articleId=article.id)


@login_required
def commentDelete(request, commentId):
    comment = get_object_or_404(Comment, id=commentId)
    article = get_object_or_404(Article, id=comment.article.id)
    if request.method == 'GET':
        return articleRead(request, article.id)

    # POST
    if comment.user != request.user:
        messages.error(request, '無刪除權限')
        return redirect('article:articleRead', articleId=article.id)

    comment.delete()
    return redirect('article:articleRead', articleId=article.id)
