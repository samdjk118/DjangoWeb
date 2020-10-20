from populate import base
from account.models import User
from article.models import Article, Comment


titles = ['如何賣雞排', '做出一杯好喝的珍奶', '簡單學習泡奶粉']
comments = ['說的真棒', '真D爛', '推推推', '阿姨，我想放棄了']


def populate():
    print('Populating articles and comments ...', end=' ')
    Article.objects.all().delete()
    Comment.objects.all().delete()
    admin = User.objects.get(username='admin')
    print(admin)
    for title in titles:
        article = Article()
        article.title = title
        for j in range(20):
            article.content += title + '\n'
        article.save()
        for comment in comments:
            Comment.objects.create(
                article=article, user=admin, content=comment)
    print('done')


if __name__ == '__main__':
    populate()
