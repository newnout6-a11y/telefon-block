"""
Сбор легитимных номеров РФ для SpamBlocker.

Источники сгруппированы по уверенности и категории:
  org (high confidence):
    zoon.ru, spravker.ru, rusprofile.ru, mosgorzdrav.ru, mos.ru
  delivery (high confidence):
    eda.yandex, lavka, samokat, sbermarket, dostavista, cdek, boxberry,
    dpd, pecom, delovye-linii, kuper
  freelancer / private (medium):
    fl.ru, hands.ru, freelance.ru, profi.ru, youdo, remontnik, drom, cian
  classified (low):
    irr.ru, farpost.ru, barahla.net
  background (very low): обычные мобильные из плана нумерации
    включается явно через --add-user-numbers

Output: datasets/ru/raw/legitimate_numbers.csv
Format: normalized_number,name,category,source,city,url,source_confidence

Usage:
    python scripts/ru_legitimate_collector.py --profile smart
    python scripts/ru_legitimate_collector.py --profile broad --max-urls 5000
    python scripts/ru_legitimate_collector.py --profile org
    python scripts/ru_legitimate_collector.py --add-user-numbers 2000
"""

import argparse
import asyncio
import csv
import html as html_lib
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, TypeVar
from urllib.parse import quote, unquote, urlparse, urljoin

try:
    import aiohttp
except ImportError:
    print("Требуется aiohttp:  pip install aiohttp")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(__file__))
from ru_number_normalizer import normalize_ru_phone, is_russian_number

# ── Config ──────────────────────────────────────────────────────────────────

CONCURRENCY = 20
DELAY_MIN = 0.05
DELAY_MAX = 0.3
MAX_PAGES = 30
OUTPUT_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', 'datasets', 'ru', 'raw', 'legitimate_numbers.csv'
))

# 2GIS Catalog API key. Public free-tier: ~1k requests/day without key for the
# `m.2gis.ru` mobile catalog; with a registered key (free) ~50k/day. Set via
# environment variable; the collector silently skips 2GIS if missing.
TWO_GIS_API_KEY = os.environ.get('TWO_GIS_API_KEY') or os.environ.get('DGIS_API_KEY') or ''
STATE_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', 'datasets', 'ru', 'raw', 'legitimate_collector_state.json'
))

# ── Spravker: city subdomains (tested working) ─────────────────────────────

SPRAVKER_CITIES = {
    # Tier 1: 16 original cities (tested working)
    'msk': 'msk.spravker.ru', 'spb': 'spb.spravker.ru',
    'ekb': 'ekaterinburg.spravker.ru', 'kzn': 'kazan.spravker.ru',
    'nnov': 'nizhnij-novgorod.spravker.ru', 'rnd': 'rostov-na-donu.spravker.ru',
    'ufa': 'ufa.spravker.ru', 'krasnodar': 'krasnodar.spravker.ru',
    'voronezh': 'voronezh.spravker.ru', 'chelyabinsk': 'chelyabinsk.spravker.ru',
    'samara': 'samara.spravker.ru', 'perm': 'perm.spravker.ru',
    'volgograd': 'volgograd.spravker.ru', 'barnaul': 'barnaul.spravker.ru',
    'tyumen': 'tyumen.spravker.ru',
    'novgorod': 'novgorod.spravker.ru',
    # Tier 2: extension to 50 — major regional centers (>300k pop)
    'novosibirsk': 'novosib.spravker.ru',
    'omsk': 'omsk.spravker.ru',
    'krasnoyarsk': 'krasnoyarsk.spravker.ru',
    'vladivostok': 'vladivostok.spravker.ru',
    'sochi': 'sochi.spravker.ru',
    'kaliningrad': 'kaliningrad.spravker.ru',
    'saratov': 'saratov.spravker.ru',
    'izhevsk': 'izhevsk.spravker.ru',
    'ulyanovsk': 'ulyanovsk.spravker.ru',
    'yaroslavl': 'yar.spravker.ru',
    'khabarovsk': 'khabarovsk.spravker.ru',
    'irkutsk': 'irkutsk.spravker.ru',
    'tula': 'tula.spravker.ru',
    'kursk': 'kursk.spravker.ru',
    'kirov': 'kirov.spravker.ru',
    'lipetsk': 'lipetsk.spravker.ru',
    'penza': 'penza.spravker.ru',
    'astrakhan': 'astrahan.spravker.ru',
    'tomsk': 'tomsk.spravker.ru',
    'ryazan': 'ryazan.spravker.ru',
    'cheboksary': 'cheboksary.spravker.ru',
    'magnitogorsk': 'magnitogorsk.spravker.ru',
    'ivanovo': 'ivanovo.spravker.ru',
    'vladimir': 'vladimir.spravker.ru',
    'belgorod': 'belgorod.spravker.ru',
    'murmansk': 'murmansk.spravker.ru',
    'stavropol': 'stavropol.spravker.ru',
    'kemerovo': 'kemerovo.spravker.ru',
    'orel': 'orel.spravker.ru',
    'smolensk': 'smolensk.spravker.ru',
    'kaluga': 'kaluga.spravker.ru',
    'tambov': 'tambov.spravker.ru',
    'orenburg': 'orenburg.spravker.ru',
    'nalchik': 'nalchik.spravker.ru',
    'makhachkala': 'makhachkala.spravker.ru',
    'abakan': 'abakan.spravker.ru',
    'surgut': 'surgut.spravker.ru',
    'arhangelsk': 'arhangelsk.spravker.ru',
    'vologda': 'vologda.spravker.ru',
    'taganrog': 'taganrog.spravker.ru',
    'chita': 'chita.spravker.ru',
    # Tier 3: Phase-1 ALLOW ×10 expansion — cities with population
    # 100k-300k. spravker.ru hosts subdomains for ~150 RU cities total;
    # this batch covers the next 80 most populated regional centers and
    # major district capitals.
    'novokuznetsk': 'novokuznetsk.spravker.ru',
    'tolyatti': 'tolyatti.spravker.ru',
    'naberezhnye-chelny': 'naberezhnye-chelny.spravker.ru',
    'cherepovets': 'cherepovets.spravker.ru',
    'sterlitamak': 'sterlitamak.spravker.ru',
    'nizhnevartovsk': 'nizhnevartovsk.spravker.ru',
    'novorossiysk': 'novorossiysk.spravker.ru',
    'yakutsk': 'yakutsk.spravker.ru',
    'orsk': 'orsk.spravker.ru',
    'volzhsky': 'volzhskij.spravker.ru',
    'pskov': 'pskov.spravker.ru',
    'syzran': 'syzran.spravker.ru',
    'kovrov': 'kovrov.spravker.ru',
    'staryy-oskol': 'staryj-oskol.spravker.ru',
    'engels': 'engels.spravker.ru',
    'balashikha': 'balashikha.spravker.ru',
    'podolsk': 'podolsk.spravker.ru',
    'khimki': 'himki.spravker.ru',
    'mytischi': 'mytischi.spravker.ru',
    'korolev': 'korolev.spravker.ru',
    'lyubertsy': 'lyubertsy.spravker.ru',
    'krasnogorsk': 'krasnogorsk.spravker.ru',
    'odintsovo': 'odintsovo.spravker.ru',
    'elektrostal': 'elektrostal.spravker.ru',
    'kolomna': 'kolomna.spravker.ru',
    'serpukhov': 'serpuhov.spravker.ru',
    'pushkino': 'pushkino.spravker.ru',
    'reutov': 'reutov.spravker.ru',
    'domodedovo': 'domodedovo.spravker.ru',
    'shchelkovo': 'schelkovo.spravker.ru',
    'sergiev-posad': 'sergiev-posad.spravker.ru',
    'noginsk': 'noginsk.spravker.ru',
    'orekhovo': 'orehovo-zuevo.spravker.ru',
    'ramenskoye': 'ramenskoe.spravker.ru',
    'dolgoprudny': 'dolgoprudnyj.spravker.ru',
    'zhukovsky': 'zhukovskij.spravker.ru',
    'gatchina': 'gatchina.spravker.ru',
    'vyborg': 'vyborg.spravker.ru',
    'tikhvin': 'tikhvin.spravker.ru',
    'severodvinsk': 'severodvinsk.spravker.ru',
    'velikiy-novgorod': 'velikij-novgorod.spravker.ru',
    'petrozavodsk': 'petrozavodsk.spravker.ru',
    'syktyvkar': 'syktyvkar.spravker.ru',
    'ukhta': 'uhta.spravker.ru',
    'salekhard': 'salehard.spravker.ru',
    'noyabrsk': 'noyabrsk.spravker.ru',
    'novyy-urengoy': 'novyj-urengoj.spravker.ru',
    'nadym': 'nadym.spravker.ru',
    'nefteyugansk': 'nefteyugansk.spravker.ru',
    'kogalym': 'kogalym.spravker.ru',
    'tobolsk': 'tobolsk.spravker.ru',
    'kurgan': 'kurgan.spravker.ru',
    'shadrinsk': 'shadrinsk.spravker.ru',
    'miass': 'miass.spravker.ru',
    'zlatoust': 'zlatoust.spravker.ru',
    'kopeysk': 'kopejsk.spravker.ru',
    'kamensk-uralsky': 'kamensk-uralskij.spravker.ru',
    'pervouralsk': 'pervouralsk.spravker.ru',
    'nizhny-tagil': 'nizhnij-tagil.spravker.ru',
    'serov': 'serov.spravker.ru',
    'berezniki': 'berezniki.spravker.ru',
    'solikamsk': 'solikamsk.spravker.ru',
    'salavat': 'salavat.spravker.ru',
    'oktyabrsky': 'oktyabrskij.spravker.ru',
    'neftekamsk': 'neftekamsk.spravker.ru',
    'almetyevsk': 'almetevsk.spravker.ru',
    'nizhnekamsk': 'nizhnekamsk.spravker.ru',
    'bugulma': 'bugulma.spravker.ru',
    'elabuga': 'elabuga.spravker.ru',
    'arzamas': 'arzamas.spravker.ru',
    'dzerzhinsk': 'dzerzhinsk.spravker.ru',
    'sarov': 'sarov.spravker.ru',
    'cheboksary-2': 'novocheboksarsk.spravker.ru',
    'yoshkar-ola': 'joshkar-ola.spravker.ru',
    'saransk': 'saransk.spravker.ru',
    'kostroma': 'kostroma.spravker.ru',
    'rybinsk': 'rybinsk.spravker.ru',
    'pereslavl': 'pereslavl.spravker.ru',
    'tver': 'tver.spravker.ru',
    'rzhev': 'rzhev.spravker.ru',
    'bryansk': 'bryansk.spravker.ru',
    'kursk-2': 'zheleznogorsk.spravker.ru',
    'voronezh-2': 'borisoglebsk.spravker.ru',
    'tambov-2': 'michurinsk.spravker.ru',
    'lipetsk-2': 'elec.spravker.ru',
    'volgograd-2': 'volzhsky-vlg.spravker.ru',
    'sochi-2': 'novorossiysk-2.spravker.ru',
    'krasnodar-2': 'armavir.spravker.ru',
    'krasnodar-3': 'maikop.spravker.ru',
    'rostov-2': 'shahty.spravker.ru',
    'rostov-3': 'novocherkassk.spravker.ru',
    'rostov-4': 'volgodonsk.spravker.ru',
    'stavropol-2': 'pyatigorsk.spravker.ru',
    'stavropol-3': 'kislovodsk.spravker.ru',
    'stavropol-4': 'essentuki.spravker.ru',
    'stavropol-5': 'mineralnyje-vody.spravker.ru',
    'vladikavkaz': 'vladikavkaz.spravker.ru',
    'grozny': 'groznyj.spravker.ru',
    'kazan-2': 'zelenodolsk.spravker.ru',
    'simferopol': 'simferopol.spravker.ru',
    'sevastopol': 'sevastopol.spravker.ru',
    'kerch': 'kerch.spravker.ru',
    'evpatoria': 'evpatoriya.spravker.ru',
    'yalta': 'yalta.spravker.ru',
    'feodosia': 'feodosiya.spravker.ru',
    'krasnoyarsk-2': 'achinsk.spravker.ru',
    'krasnoyarsk-3': 'norilsk.spravker.ru',
    'krasnoyarsk-4': 'minusinsk.spravker.ru',
    'irkutsk-2': 'angarsk.spravker.ru',
    'irkutsk-3': 'bratsk.spravker.ru',
    'ulan-ude': 'ulan-ude.spravker.ru',
    'kyzyl': 'kyzyl.spravker.ru',
    'gorno-altaysk': 'gorno-altajsk.spravker.ru',
    'biysk': 'bijsk.spravker.ru',
    'rubtsovsk': 'rubcovsk.spravker.ru',
    'novosibirsk-2': 'berdsk.spravker.ru',
    'kemerovo-2': 'leninsk-kuzneckij.spravker.ru',
    'kemerovo-3': 'mezhdurechensk.spravker.ru',
    'tomsk-2': 'seversk.spravker.ru',
    'omsk-2': 'tara.spravker.ru',
    'blagoveshchensk': 'blagoveschensk.spravker.ru',
    'komsomolsk-na-amure': 'komsomolsk-na-amure.spravker.ru',
    'birobidzhan': 'birobidzhan.spravker.ru',
    'magadan': 'magadan.spravker.ru',
    'petropavlovsk-kamchatsky': 'petropavlovsk-kamchatskij.spravker.ru',
    'yuzhno-sakhalinsk': 'juzhno-sahalinsk.spravker.ru',
    'anadyr': 'anadyr.spravker.ru',
    'norilsk': 'norilsk.spravker.ru',
    'naryan-mar': 'naryan-mar.spravker.ru',
    'kandalaksha': 'kandalaksha.spravker.ru',
    'apatity': 'apatity.spravker.ru',
}

# Spravker subcategories that have phones in listing pages (tested)
SPRAVKER_SUBCATEGORIES = [
    'bolnicy', 'stomatologicheskie-kliniki-i-tsentryi',
    'stomatologicheskie-polikliniki', 'medtsentryi-i-kliniki',
    'diagnosticheskie-tsentryi', 'apteki',
    'banki', 'strahovanye-kompanii',
    'universitetyi', 'shkolyi',
    'avtoservisyi', 'avtosalonyi',
    'notariusyi', 'advokatskije-kollegii',
    'turisticheskiye-agentstva', 'gostinitsyi',
    'salonyi-krasotyi', 'parikmaherskiye',
    'fitnes-klubyi', 'sportivnyje-klubyi',
    'magazinyi-produktov', 'supermarketyi',
    'restoranyi', 'kafe',
    'remont-kvartir', 'okna',
    'agentstva-nedvizhimosti', 'rieltorskiye-agentstva',
    'logisticheskiye-kompanii', 'gruzoperevozki',
    'internet-provaideryi', 'mobilnyje-operatoryi',
    'veterinarnyje-kliniki', 'detskiye-sadyi',
]

# ── Zoon: cities and categories ────────────────────────────────────────────

ZOON_CITIES = {
    'msk': 'msk', 'spb': 'spb', 'ekb': 'ekb',
    'ufa': 'ufa', 'krasnodar': 'krasnodar', 'voronezh': 'voronezh',
    'chelyabinsk': 'chelyabinsk', 'samara': 'samara', 'perm': 'perm',
    'volgograd': 'volgograd', 'barnaul': 'barnaul', 'tyumen': 'tyumen',
    'krasnoyarsk': 'krasnoyarsk', 'izhevsk': 'izhevsk', 'yaroslavl': 'yaroslavl',
    'ryazan': 'ryazan', 'tula': 'tula', 'omsk': 'omsk',
    'vladivostok': 'vladivostok', 'nizhnevartovsk': 'nizhnevartovsk',
    'saratov': 'saratov', 'ulyanovsk': 'ulyanovsk', 'irkutsk': 'irkutsk',
    'smolensk': 'smolensk', 'kaliningrad': 'kaliningrad', 'kursk': 'kursk',
    'belgorod': 'belgorod', 'lipetsk': 'lipetsk', 'bryansk': 'bryansk',
    'vladimir': 'vladimir', 'ivanovo': 'ivanovo', 'kostroma': 'kostroma',
    # Phase-1 ALLOW ×10 expansion: zoon.ru hosts subdomain catalogs for
    # most major RU cities. Adding 40 more major regional centers.
    'novosibirsk': 'novosibirsk', 'kazan': 'kazan',
    'nnov': 'nnovgorod', 'rnd': 'rostovnadonu',
    'novokuznetsk': 'novokuznetsk', 'tolyatti': 'tolyatti',
    'cherepovets': 'cherepovets', 'naberezhnye-chelny': 'naberezhnyechelny',
    'sterlitamak': 'sterlitamak', 'novorossiysk': 'novorossiysk',
    'sochi': 'sochi', 'kaluga': 'kaluga',
    'tambov': 'tambov', 'penza': 'penza', 'orel': 'orel',
    'orenburg': 'orenburg', 'magnitogorsk': 'magnitogorsk',
    'astrakhan': 'astrahan', 'tomsk': 'tomsk',
    'kemerovo': 'kemerovo', 'kaliningrad-2': 'kaliningradskaya',
    'cheboksary': 'cheboksary', 'novgorod': 'velikiynovgorod',
    'pskov': 'pskov', 'arhangelsk': 'arhangelsk',
    'syktyvkar': 'syktyvkar', 'petrozavodsk': 'petrozavodsk',
    'murmansk': 'murmansk', 'vologda': 'vologda',
    'tver': 'tver', 'kirov': 'kirov',
    'saransk': 'saransk', 'yoshkar-ola': 'yoshkarola',
    'taganrog': 'taganrog', 'pyatigorsk': 'pyatigorsk',
    'stavropol': 'stavropol', 'makhachkala': 'mahachkala',
    'simferopol': 'simferopol', 'sevastopol': 'sevastopol',
    'khabarovsk': 'habarovsk', 'blagoveshchensk': 'blagoveshchensk',
}

ZOON_CATEGORIES = [
    'medical', 'beauty', 'fitness', 'education', 'home',
    'pets', 'finance', 'legal', 'travel', 'auto',
    'food', 'children', 'entertainment', 'photo', 'music',
    'it', 'marketing', 'design', 'repair', 'cleaning',
]

# ── Rusprofile: search queries ────────────────────────────────────────────

RUSPROFILE_QUERIES = [
    'клиника', 'стоматология', 'больница', 'аптека',
    'школа', 'университет', 'детский сад', 'колледж',
    'автосервис', 'автомойка', 'шиномонтаж', 'автосалон',
    'страховая компания', 'банк отделение',
    'нотариус', 'адвокат', 'юридическая компания',
    'салон красоты', 'парикмахерская', 'барбершоп',
    'ресторан', 'кафе', 'пиццерия', 'столовая',
    'строительная компания', 'ремонт квартир', 'электрик', 'сантехник',
    'риелтор', 'агентство недвижимости', 'застройщик',
    'ветеринарная клиника', 'зоомагазин',
    'фитнес клуб', 'спортивный клуб', 'бассейн',
    'доставка еды', 'логистическая компания', 'такси',
    'супермаркет', 'магазин продуктов', 'аптека',
    'интернет провайдер', 'телекоммуникации',
    'гостиница', 'хостел', 'туристическое агентство',
    'жкх', 'управляющая компания', 'мфц',
    'поликлиника', 'диагностический центр', 'лаборатория',
    'окна установка', 'мебель на заказ', 'кухни на заказ',
    'клининг', 'прачечная', 'химчистка',
    'кредит', 'микрофинансовая', 'ломбард',
    'церковь', 'мечеть', 'храм',
]

SMART_ZOON_CATEGORIES = [
    'home', 'repair', 'cleaning', 'it', 'marketing', 'design',
    'auto', 'finance', 'legal', 'travel', 'pets', 'children',
]

SMART_SPRAVKER_SUBCATEGORIES = [
    'avtoservisyi', 'avtosalonyi',
    'notariusyi', 'advokatskije-kollegii',
    'turisticheskiye-agentstva', 'gostinitsyi',
    'salonyi-krasotyi', 'parikmaherskiye',
    'fitnes-klubyi', 'sportivnyje-klubyi',
    'magazinyi-produktov', 'supermarketyi',
    'remont-kvartir', 'okna',
    'agentstva-nedvizhimosti', 'rieltorskiye-agentstva',
    'logisticheskiye-kompanii', 'gruzoperevozki',
    'internet-provaideryi', 'mobilnyje-operatoryi',
    'veterinarnyje-kliniki',
]

DELIVERY_PUBLIC_URLS = [
    ('https://eda.yandex.ru/', 'Яндекс Еда', 'delivery'),
    ('https://eda.yandex.ru/contacts', 'Яндекс Еда', 'delivery'),
    ('https://lavka.yandex.ru/', 'Яндекс Лавка', 'delivery'),
    ('https://samokat.ru/', 'Самокат', 'delivery'),
    ('https://kuper.ru/', 'Купер', 'delivery'),
    ('https://sbermarket.ru/', 'СберМаркет', 'delivery'),
    ('https://dostavista.ru/', 'Достависта', 'delivery'),
    ('https://www.cdek.ru/ru/contacts', 'СДЭК', 'delivery'),
    ('https://www.cdek.ru/ru/offices', 'СДЭК офисы', 'delivery'),
    ('https://boxberry.ru/contacts', 'Boxberry', 'delivery'),
    ('https://boxberry.ru/find-an-office/', 'Boxberry офисы', 'delivery'),
    ('https://www.dpd.ru/', 'DPD', 'delivery'),
    ('https://www.dpd.ru/ols/contact.do2', 'DPD контакты', 'delivery'),
    ('https://pecom.ru/contacts/', 'ПЭК', 'delivery'),
    ('https://pecom.ru/services/', 'ПЭК услуги', 'delivery'),
    ('https://www.delovye-linii.ru/contacts/', 'Деловые линии', 'delivery'),
    ('https://www.delovye-linii.ru/offices/', 'Деловые линии офисы', 'delivery'),
    ('https://www.pochta.ru/support', 'Почта России', 'delivery'),
    ('https://www.pochta.ru/offices', 'Почта России офисы', 'delivery'),
    ('https://www.dellin.ru/contacts/', 'Деловые линии Деллин', 'delivery'),
    ('https://strazhcourier.ru/', 'Страж курьер', 'delivery'),
    ('https://www.energogaz.com/contacts/', 'Энергогаз доставка', 'delivery'),
    ('https://www.gett.com/ru/cities/', 'Gett такси', 'delivery'),
    ('https://citymobil.ru/', 'Ситимобил', 'delivery'),
    ('https://taximaxim.ru/', 'Максим такси', 'delivery'),
]

# ── Источники бизнес-номеров с высокой уверенностью ───────────────────────
#
# URL-наборы для крупных РФ-организаций, чьи контактные страницы публикуют
# горячие линии, региональные офисы и call-центры в виде статичного HTML с
# `tel:`-ссылками. Все они отдают валидные RU-номера без JS-рендера и без
# серьёзного антибота — проверено локально.
#
# Каждая запись: (url, display_name, category). Дальнейшие категории
# отображаются на feature-флаги через CATEGORY_KEYWORDS, поэтому
# подбираются совпадающие.

RU_BANK_CONTACT_URLS = [
    ('https://www.gazprombank.ru/personal/feedback/contacts/', 'Газпромбанк контакты', 'bank'),
    ('https://www.psbank.ru/About/ContactInformation', 'Промсвязьбанк контакты', 'bank'),
    ('https://www.rshb.ru/contacts/', 'Россельхозбанк контакты', 'bank'),
    ('https://www.rosbank.ru/about/contacts/', 'Росбанк контакты', 'bank'),
    ('https://www.tbank.ru/about/contacts/', 'Т-Банк контакты', 'bank'),
    ('https://www.raiffeisen.ru/about/contact/', 'Райффайзенбанк контакты', 'bank'),
    ('https://www.gazprombank.ru/personal/feedback/', 'Газпромбанк обратная связь', 'bank'),
    ('https://mkb.ru/about/contacts', 'МКБ контакты', 'bank'),
]

RU_TELECOM_CONTACT_URLS = [
    ('https://moscow.megafon.ru/help/contacts/', 'МегаФон контакты', 'telecom'),
    ('https://www.megafon.ru/help/contacts/', 'МегаФон контакты РФ', 'telecom'),
    ('https://www.mts.ru/personal/podderzhka/uznai-vse-o-mts/kontaktnaya-informatsiya', 'МТС контакты', 'telecom'),
    ('https://moskva.mts.ru/personal/podderzhka/uznai-vse-o-mts/kontaktnaya-informatsiya', 'МТС Москва контакты', 'telecom'),
    ('https://moskva.beeline.ru/customers/help/', 'Билайн помощь', 'telecom'),
    ('https://www.beeline.ru/customers/help/contact-info/', 'Билайн контакты', 'telecom'),
    ('https://msk.tele2.ru/help/contacts', 'Tele2 Москва контакты', 'telecom'),
]

RU_AIRLINE_CONTACT_URLS = [
    ('https://www.rossiya-airlines.com/about/contacts/', 'А/к Россия контакты', 'transport'),
    ('https://www.rossiya-airlines.com/about/contacts/representations/', 'А/к Россия офисы', 'transport'),
    ('https://www.pobeda.aero/about/contacts', 'Победа контакты', 'transport'),
    ('https://www.flyredwings.com/contacts', 'Red Wings контакты', 'transport'),
    ('https://www.utair.ru/help/contacts/', 'ЮТэйр контакты', 'transport'),
    ('https://www.utair.ru/help/representative_offices/', 'ЮТэйр офисы', 'transport'),
    ('https://flysmartavia.com/contacts/', 'Smartavia контакты', 'transport'),
]

RU_AIRPORT_CONTACT_URLS = [
    ('https://pulkovoairport.ru/contacts/', 'Пулково контакты', 'transport'),
    ('https://www.svo.aero/ru/contacts/', 'Шереметьево контакты', 'transport'),
    ('https://www.vnukovo.ru/contacts/', 'Внуково контакты', 'transport'),
    ('https://www.dme.ru/contacts/', 'Домодедово контакты', 'transport'),
    ('https://airportufa.ru/passengers/contacts/', 'Аэропорт Уфа контакты', 'transport'),
    ('https://koltsovo.ru/about/contacts/', 'Кольцово (Екб) контакты', 'transport'),
]

RU_PRESS_CONTACT_URLS = [
    ('https://aif.ru/contacts', 'АиФ контакты', 'business'),
    ('https://www.kp.ru/site/contacts.html', 'Комсомольская правда контакты', 'business'),
    ('https://www.kommersant.ru/about/contacts', 'Коммерсантъ контакты', 'business'),
    ('https://www.vedomosti.ru/info/contact', 'Ведомости контакты', 'business'),
    ('https://lenta.ru/info/posts/contacts/', 'Лента.ру контакты', 'business'),
    ('https://ria.ru/contacts/', 'РИА Новости контакты', 'business'),
]

RU_UNIVERSITY_CONTACT_URLS = [
    ('https://www.hse.ru/contacts', 'ВШЭ контакты', 'education'),
    ('https://itmo.ru/ru/contact_info.htm', 'ИТМО контакты', 'education'),
    ('https://bmstu.ru/contacts', 'МГТУ Баумана контакты', 'education'),
    ('https://www.rudn.ru/about/contacts', 'РУДН контакты', 'education'),
    ('https://mipt.ru/about/contacts/', 'МФТИ контакты', 'education'),
    ('https://mgimo.ru/about/contacts/', 'МГИМО контакты', 'education'),
    ('https://spbu.ru/o-spbgu/struktura/kontakty', 'СПбГУ контакты', 'education'),
    ('https://www.msu.ru/info/struct/', 'МГУ структура', 'education'),
]

RU_INSURANCE_CONTACT_URLS = [
    ('https://www.rgs.ru/about/contacts/', 'Росгосстрах контакты', 'insurance'),
    ('https://www.rgs.ru/about/contacts/index.wbp', 'Росгосстрах контакты wbp', 'insurance'),
    ('https://sogaz.ru/about/contacts/', 'СОГАЗ контакты', 'insurance'),
    ('https://www.renins.ru/contacts/', 'Ренессанс Страхование контакты', 'insurance'),
    ('https://www.soglasie.ru/about/contacts/', 'Согласие контакты', 'insurance'),
    ('https://www.ingos.ru/company/contacts/', 'Ингосстрах контакты', 'insurance'),
    ('https://www.vsk.ru/about/contacts/', 'ВСК контакты', 'insurance'),
    ('https://www.alfastrah.ru/contacts/', 'АльфаСтрахование контакты', 'insurance'),
]

RU_RETAIL_CONTACT_URLS = [
    ('https://magnit.ru/about/contacts', 'Магнит контакты', 'retail'),
    ('https://vkusvill.ru/contacts/', 'ВкусВилл контакты', 'retail'),
    ('https://www.metro-cc.ru/contacts', 'Metro Cash&Carry контакты', 'retail'),
    ('https://dixy.ru/about/contacts/', 'Дикси контакты', 'retail'),
    ('https://www.lamoda.ru/info/contacts/', 'Lamoda контакты', 'retail'),
    ('https://www.okmarket.ru/contacts/', 'О’Кей контакты', 'retail'),
    ('https://lenta.com/contacts/', 'Лента контакты', 'retail'),
    ('https://www.perekrestok.ru/contacts', 'Перекрёсток контакты', 'retail'),
]

RU_FEDERAL_HOTLINE_URLS = [
    # Только страницы, которые в локальных тестах гарантированно отвечают за <10 с
    # и отдают валидные федеральные горячие линии. rkn/fssp/rospotrebnadzor/rosreestr
    # из этого списка исключены — они стабильно зависают на CDN или таймаутят.
    ('https://sfr.gov.ru/contacts/', 'СФР контакты', 'government'),
    ('https://www.nalog.gov.ru/rn77/about_fts/single_tel/', 'ФНС горячие линии', 'government'),
    ('https://mchs.gov.ru/kontakty', 'МЧС контакты', 'government'),
]

# Источники маркетплейсов и крупных e-commerce — поддержка/контакты публичные
RU_MARKETPLACE_CONTACT_URLS = [
    ('https://www.lamoda.ru/info/contacts/', 'Lamoda контакты', 'business'),
    ('https://yandex.ru/support/market/contact-us.html', 'Яндекс Маркет поддержка', 'business'),
    ('https://help.mail.ru/', 'Mail.ru поддержка', 'business'),
]

SERVICE_MARKETPLACE_URLS = [
    ('https://www.fl.ru/freelancers/', 'FL.ru', 'freelancer'),
    ('https://www.fl.ru/freelancers/programmer/', 'FL.ru программисты', 'freelancer'),
    ('https://www.fl.ru/freelancers/design/', 'FL.ru дизайнеры', 'freelancer'),
    ('https://www.fl.ru/freelancers/copywriting/', 'FL.ru копирайтеры', 'freelancer'),
    ('https://www.fl.ru/freelancers/translation/', 'FL.ru переводчики', 'freelancer'),
    ('https://www.fl.ru/projects/', 'FL.ru проекты', 'freelancer'),
    ('https://freelance.ru/', 'Freelance.ru', 'freelancer'),
    ('https://freelance.ru/projects/', 'Freelance.ru проекты', 'freelancer'),
    ('https://freelance.ru/freelancers/', 'Freelance.ru фрилансеры', 'freelancer'),
    ('https://www.remontnik.ru/', 'Remontnik.ru', 'private_seller'),
    ('https://www.remontnik.ru/masters/', 'Remontnik.ru мастера', 'private_seller'),
    ('https://www.remontnik.ru/masters/elektrik/', 'Remontnik электрики', 'private_seller'),
    ('https://www.remontnik.ru/masters/santehnik/', 'Remontnik сантехники', 'private_seller'),
    ('https://www.remontnik.ru/masters/plotnik/', 'Remontnik плотники', 'private_seller'),
    ('https://www.remontnik.ru/masters/dizayner/', 'Remontnik дизайнеры', 'private_seller'),
    ('https://profi.ru/remont/', 'Профи ремонт', 'private_seller'),
    ('https://profi.ru/repetitor/', 'Профи репетиторы', 'private_seller'),
    ('https://profi.ru/uborka/', 'Профи уборка', 'private_seller'),
    ('https://profi.ru/krasota/', 'Профи красота', 'private_seller'),
    ('https://profi.ru/transport/', 'Профи транспорт', 'private_seller'),
    ('https://profi.ru/avto/', 'Профи авто', 'private_seller'),
    ('https://youdo.com/', 'YouDo', 'private_seller'),
    ('https://youdo.com/tasks-all-opened/', 'YouDo задания', 'private_seller'),
    ('https://youdo.com/repair/', 'YouDo ремонт', 'private_seller'),
    ('https://youdo.com/cleaning/', 'YouDo уборка', 'private_seller'),
    ('https://youdo.com/courier/', 'YouDo курьеры', 'private_seller'),
    ('https://hands.ru/programmers/', 'Hands программисты', 'freelancer'),
    ('https://hands.ru/designers/', 'Hands дизайнеры', 'freelancer'),
    ('https://hands.ru/translators/', 'Hands переводчики', 'freelancer'),
    ('https://hands.ru/copywriters/', 'Hands копирайтеры', 'freelancer'),
    ('https://hands.ru/photographers/', 'Hands фотографы', 'freelancer'),
    ('https://hands.ru/composers/', 'Hands композиторы', 'freelancer'),
]

FAST_SERVICE_MARKETPLACE_URLS = [
    item for item in SERVICE_MARKETPLACE_URLS
    if 'remontnik.ru' not in item[0] and 'profi.ru' not in item[0]
]

FREELANCE_QUERIES = [
    'разработчик', 'дизайнер', 'копирайтер', 'маркетолог',
    'фотограф', 'видеограф', 'репетитор', 'переводчик',
    'сантехник', 'электрик', 'ремонт', 'курьер',
    'программист', 'верстальщик', 'таргетолог', 'smm',
    'грузчик', 'няня', 'сиделка', 'мастер на час',
]

CLASSIFIED_PUBLIC_URLS = [
    ('https://www.irr.ru/real-estate/', 'Из рук в руки недвижимость', 'private_seller'),
    ('https://www.irr.ru/real-estate/apartments-sale/', 'Из рук в руки квартиры', 'private_seller'),
    ('https://www.irr.ru/real-estate/apartments-rent/', 'Из рук в руки аренда', 'private_seller'),
    ('https://www.irr.ru/cars/used/', 'Из рук в руки авто', 'private_seller'),
    ('https://www.irr.ru/cars/new/', 'Из рук в руки новые авто', 'private_seller'),
    ('https://www.irr.ru/services/', 'Из рук в руки услуги', 'private_seller'),
    ('https://www.irr.ru/services/repair/', 'Из рук в руки ремонт', 'private_seller'),
    ('https://www.irr.ru/services/transport/', 'Из рук в руки транспорт', 'private_seller'),
    ('https://www.irr.ru/services/educational/', 'Из рук в руки образование', 'private_seller'),
    ('https://www.irr.ru/work/', 'Из рук в руки работа', 'private_seller'),
    ('https://www.farpost.ru/auto/', 'FarPost авто', 'private_seller'),
    ('https://www.farpost.ru/realty/', 'FarPost недвижимость', 'private_seller'),
    ('https://www.farpost.ru/service/', 'FarPost услуги', 'private_seller'),
    ('https://www.farpost.ru/work/', 'FarPost работа', 'private_seller'),
    ('https://barahla.net/services/', 'Barahla услуги', 'private_seller'),
    ('https://barahla.net/realty/', 'Barahla недвижимость', 'private_seller'),
    ('https://barahla.net/transport/', 'Barahla транспорт', 'private_seller'),
    ('https://barahla.net/work/', 'Barahla работа', 'private_seller'),
    ('https://www.youla.ru/all/uslugi', 'Юла услуги', 'private_seller'),
    ('https://www.youla.ru/all/transport', 'Юла транспорт', 'private_seller'),
    ('https://www.youla.ru/all/nedvizhimost', 'Юла недвижимость', 'private_seller'),
    ('https://www.avito.ru/rossiya/uslugi', 'Авито услуги', 'private_seller'),
    ('https://www.avito.ru/rossiya/transport', 'Авито транспорт', 'private_seller'),
    ('https://www.avito.ru/rossiya/nedvizhimost', 'Авито недвижимость', 'private_seller'),
]

# ── Category inference ─────────────────────────────────────────────────────

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    'bank':        ['банк', 'финанс', 'сбер', 'тинькофф', 'втб', 'альфа', 'кредит', 'ломбард', 'микрофинанс'],
    'government':  ['госуслуг', 'мфц', 'министерств', 'полиц', 'налог', 'мвд', 'жкх', 'управляющ'],
    'telecom':     ['мтс', 'билайн', 'мегафон', 'tele2', 'ростелеком', 'провайдер', 'телеком'],
    'medical':     ['больниц', 'клиник', 'стоматолог', 'аптек', 'медиц', 'госпитал', 'диагност', 'ветеринар', 'поликлиник', 'лаборатор'],
    'education':   ['университет', 'школ', 'колледж', 'институт', 'лицей', 'гимназ', 'детский сад'],
    'retail':      ['магазин', 'супермаркет', 'пятёрочк', 'магнит', 'лента', 'зоомагазин'],
    'delivery':    ['доставк', 'логист', 'перевозк', 'такси'],
    'insurance':   ['страх'],
    'transport':   ['автосервис', 'автомобил', 'автомойк', 'шиномонтаж', 'автосалон', 'гостиниц', 'хостел', 'турист'],
    'utility':     ['ремонт', 'строительн', 'электрик', 'сантехник', 'окна', 'мебель', 'кухн', 'клининг', 'прачечн', 'химчистк'],
    'legal':       ['юридич', 'нотариус', 'адвокат'],
    'realestate':  ['риелтор', 'недвижим', 'застройщик'],
    'restaurant':  ['ресторан', 'кафе', 'пиццер', 'столов'],
    'beauty':      ['салон', 'красот', 'парикмахер', 'барбершоп'],
    'fitness':     ['фитнес', 'спортивн', 'бассейн'],
    'religious':   ['церковь', 'мечеть', 'храм'],
    'other':       [],
}

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 Safari/605.1.15',
]

# ── Regex patterns ─────────────────────────────────────────────────────────

PHONE_RE = re.compile(r'(?:\+7|8)[\s\-().\xa0]*\d{3}[\s\-().\xa0]*\d{3}[\s\-().\xa0]*\d{2}[\s\-().\xa0]*\d{2}')
PHONE_COMPACT_RE = re.compile(r'(?<!\d)(?:\+?7|8)\d{10}(?!\d)')
TEL_HREF_RE = re.compile(r'''href\s*=\s*["']tel:([^"']+)["']''', re.IGNORECASE)
TITLE_RE = re.compile(r'<title[^>]*>(.*?)</title>', re.IGNORECASE | re.DOTALL)
TAG_STRIP_RE = re.compile(r'<[^>]+>')
WHITESPACE_RE = re.compile(r'\s+')
# Spravker .htm org page links
SPRAVKER_ORG_RE = re.compile(r'''href\s*=\s*["']([^"']*\.htm)["']''', re.IGNORECASE)
# Spravker pagination: each ``<a>`` anchor with class
# ``pagination-list__link`` carries an ``href="...?page=N"``. We extract
# every full anchor tag and then check it has both the class and the
# query parameter — ordering of attributes within the tag doesn't matter.
SPRAVKER_PAGINATION_RE = re.compile(r'<a\b[^>]*pagination-list__link[^>]*>', re.IGNORECASE)
SPRAVKER_PAGE_QS_RE = re.compile(r'\?page=(\d+)', re.IGNORECASE)


def _spravker_max_page(html: str) -> int:
    """Return the highest pagination page number on a spravker listing,
    or 1 if no pagination present.

    Implementation: iterate through anchor tags carrying the
    ``pagination-list__link`` class and pull ``?page=N`` from each href.
    Robust to attribute order (class before/after href) and to whitespace
    variations.
    """
    nums: List[int] = []
    for tag in SPRAVKER_PAGINATION_RE.findall(html):
        for n in SPRAVKER_PAGE_QS_RE.findall(tag):
            if n.isdigit():
                nums.append(int(n))
    return max(nums) if nums else 1
HREF_RE = re.compile(r'''href\s*=\s*["']([^"']+)["']''', re.IGNORECASE)
# Rusprofile company page links from search results
RUSPROFILE_ORG_RE = re.compile(r'href="(/id/\d+)"')
# Zoon JSON-LD
LD_JSON_RE = re.compile(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', re.DOTALL)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('legit_collector')


# ── Data ───────────────────────────────────────────────────────────────────

@dataclass
class LegitEntry:
    normalized_number: str
    name: str
    category: str
    source: str
    city: str
    url: str
    source_confidence: float = 0.70


# ── Helpers ────────────────────────────────────────────────────────────────

def infer_category(text: str) -> str:
    lower = text.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if cat == 'other':
            continue
        for kw in keywords:
            if kw in lower:
                return cat
    return 'other'


def is_plausible_phone(norm: Optional[str]) -> bool:
    if not norm or len(norm) != 12:
        return False
    if not is_russian_number(norm):
        return False
    digits = norm[2:]
    if digits == digits[0] * len(digits):
        return False
    if digits in {'1234567890', '9876543210', '0000000000'}:
        return False
    return True


def extract_phones(html: str) -> List[str]:
    """Extract and normalize Russian phone numbers from HTML."""
    results = []
    text = html_lib.unescape(html or '')
    # 1. href="tel:" links (most reliable)
    for raw in TEL_HREF_RE.findall(text):
        norm = normalize_ru_phone(unquote(raw).strip(), reject_non_ru=True)
        if is_plausible_phone(norm) and norm not in results:
            results.append(norm)
    # 2. Regex phones in text (including \xa0 non-breaking spaces)
    for raw in PHONE_RE.findall(text):
        norm = normalize_ru_phone(raw.replace('\xa0', ' '), reject_non_ru=True)
        if is_plausible_phone(norm) and norm not in results:
            results.append(norm)
    # 3. Compact numbers in JSON/JS blobs: +79001234567, 89001234567, 79001234567
    for raw in PHONE_COMPACT_RE.findall(text):
        norm = normalize_ru_phone(raw, reject_non_ru=True)
        if is_plausible_phone(norm) and norm not in results:
            results.append(norm)
    return results


_STATIC_EXT = (
    '.css', '.js', '.ico', '.svg', '.png', '.jpg', '.jpeg', '.gif', '.webp',
    '.woff', '.woff2', '.ttf', '.otf', '.eot', '.mp4', '.webm', '.mp3',
    '.pdf', '.zip', '.rar', '.xml', '.json', '.webmanifest', '.map',
)


def _looks_static(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(_STATIC_EXT)


def extract_links(html: str, base_url: str, limit: int = 25) -> List[str]:
    links: List[str] = []
    base_host = urlparse(base_url).netloc
    for raw in HREF_RE.findall(html or ''):
        href = html_lib.unescape(raw).strip()
        if not href or href.startswith(('tel:', 'mailto:', 'javascript:', '#', 'data:')):
            continue
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        if parsed.scheme not in {'http', 'https'} or parsed.netloc != base_host:
            continue
        if _looks_static(full):
            continue
        clean = full.split('#', 1)[0]
        if clean not in links:
            links.append(clean)
        if len(links) >= limit:
            break
    return links


def extract_title(html: str) -> str:
    m = TITLE_RE.search(html)
    if not m:
        return ''
    text = TAG_STRIP_RE.sub(' ', m.group(1))
    text = WHITESPACE_RE.sub(' ', text).strip()
    return text[:200]


def parse_confidence(value: str, default: float = 0.70) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def default_source_confidence(category: str, source: str) -> float:
    if source in {'official_whitelist', 'official_hotline'}:
        return 0.95
    if category == 'personal_mobile' or source.startswith('numbering_plan'):
        return 0.25
    if category in {'private_seller', 'realestate_owner'}:
        return 0.55
    if category in {'freelancer', 'realestate_agent'}:
        return 0.60
    if category in {'delivery', 'government', 'bank', 'medical'}:
        return 0.85
    return 0.70


# ── Async HTTP ─────────────────────────────────────────────────────────────

class AsyncScraper:
    def __init__(self, concurrency: int = CONCURRENCY):
        self.sem = asyncio.Semaphore(concurrency)
        self.session: Optional[aiohttp.ClientSession] = None
        self.seen: Set[str] = set()
        self.results: List[LegitEntry] = []
        self.host_locks: Dict[str, asyncio.Lock] = {}
        self.host_last: Dict[str, float] = {}
        self.visited_urls: Set[str] = set()  # skip already-fetched URLs
        # Per-URL last-fetch timestamps (epoch seconds) so save_state can
        # purge entries older than VISITED_TTL_S on the next load instead
        # of letting the dedup set grow forever or be wiped wholesale every
        # iter (the previous workflow's behaviour).
        self.visited_url_ts: Dict[str, int] = {}
        self._resumed_fetched: int = 0  # fetched count from previous runs
        self.stats = {'fetched': 0, 'failed': 0, 'phones_found': 0, 'skipped': 0}
        self.blacklist: Set[str] = set()  # numbers from fraud/suspect databases

    async def start(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=20, sock_read=12),
            headers={'Accept-Language': 'ru-RU,ru;q=0.9,en;q=0.5'}
        )

    async def close(self):
        if self.session:
            await self.session.close()

    def _get_host_lock(self, host: str) -> asyncio.Lock:
        if host not in self.host_locks:
            self.host_locks[host] = asyncio.Lock()
        return self.host_locks[host]

    async def fetch(self, url: str, allow_status: Set[int] = None) -> Optional[str]:
        # Skip already visited
        if url in self.visited_urls:
            self.stats['skipped'] += 1
            return None
        # URL limit check (count only new fetches this session)
        max_urls = getattr(self, '_max_urls', 0)
        new_fetched = self.stats['fetched'] - self._resumed_fetched
        if max_urls > 0 and new_fetched >= max_urls:
            return None
        # Progress counter
        new_fetched = self.stats['fetched'] - self._resumed_fetched + 1
        if max_urls > 0:
            log.info(f"[{new_fetched}/{max_urls}] {url}")
        elif new_fetched % 50 == 0:
            log.info(f"[{new_fetched}] fetched, {len(self.results)} numbers")
        from urllib.parse import urlparse
        host = urlparse(url).netloc
        lock = self._get_host_lock(host)
        self.visited_urls.add(url)
        self.visited_url_ts[url] = int(time.time())

        async with lock:
            now = time.monotonic()
            elapsed = now - self.host_last.get(host, 0)
            min_delay = DELAY_MIN if 'rusprofile' not in host else 0.8
            if elapsed < min_delay:
                await asyncio.sleep(min_delay - elapsed)

            headers = {'User-Agent': random.choice(USER_AGENTS)}
            allow = allow_status or {200}

            for attempt in range(3):
                try:
                    async with self.sem:
                        async with self.session.get(url, headers=headers, ssl=False) as resp:
                            self.host_last[host] = time.monotonic()
                            if resp.status in allow:
                                text = await resp.text(errors='replace')
                                self.stats['fetched'] += 1
                                return text
                            elif resp.status in {404, 410}:
                                return None
                            else:
                                self.stats['failed'] += 1
                                return None
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    if attempt < 2:
                        await asyncio.sleep(1.5 * (attempt + 1))
                    else:
                        self.stats['failed'] += 1
                        return None
            return None

    def load_blacklist(self):
        """Load suspect/fraud numbers from ru_reputation_raw.csv."""
        csv_path = os.path.normpath(os.path.join(
            os.path.dirname(__file__), '..', 'datasets', 'ru', 'raw', 'ru_reputation_raw.csv'
        ))
        if not os.path.exists(csv_path):
            return
        with open(csv_path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                n = row.get('normalized_number', '').strip()
                if n:
                    self.blacklist.add(n)
        log.info(f"  Loaded {len(self.blacklist)} suspect numbers from reputation_raw")

    def add(self, entry: LegitEntry) -> bool:
        key = entry.normalized_number
        if key in self.seen:
            return False
        # Block known fraud/suspect numbers (unless from official_whitelist)
        if key in self.blacklist and entry.source != 'official_whitelist' and entry.source != 'official_hotline':
            return False
        self.seen.add(key)
        self.results.append(entry)
        return True

    def add_phones(self, phones: List[str], name: str, category: str,
                   source: str, city: str, url: str, source_confidence: float = 0.70) -> int:
        added = 0
        confidence = source_confidence if source_confidence != 0.70 else default_source_confidence(category, source)
        for phone in phones:
            if self.add(LegitEntry(phone, name, category, source, city, url, confidence)):
                added += 1
        self.stats['phones_found'] += len(phones)
        return added


# ── Mid-source checkpoint helper ──────────────────────────────────────────
# Long fan-out sources routinely overshoot the per-iter timeout, which kills
# the process before the post-source `save_results` runs. Checkpoint mid-source
# so partial progress survives SIGTERM.

T = TypeVar('T')


def _checkpoint(scraper: 'AsyncScraper') -> None:
    out = getattr(scraper, '_output_path', None)
    state = getattr(scraper, '_state_path', None)
    if out:
        try:
            save_results(scraper.results, out)
        except Exception as e:  # checkpoint must never crash the loop
            log.warning(f"  checkpoint save_results failed: {e}")
    if state and hasattr(scraper, 'visited_urls'):
        try:
            save_state(scraper, state)
        except Exception as e:
            log.warning(f"  checkpoint save_state failed: {e}")


def _shuffle_sample(items: List[T], limit: int) -> List[T]:
    if limit <= 0:
        return []
    sample = list(items)
    random.shuffle(sample)
    return sample[:limit]


# ── Source 1: zoon.ru ──────────────────────────────────────────────────────

async def scrape_zoon(scraper: AsyncScraper, cities: Dict[str, str],
                      categories: List[str]) -> int:
    """zoon.ru — JSON-LD ItemList gives 30 named orgs + 31 tel: links per page.
    No pagination (same HTML for ?page=N), so we vary categories × cities."""
    total = 0
    for city_key, city_slug in cities.items():
        city_added = 0
        for cat in categories:
            url = f'https://zoon.ru/{city_slug}/{cat}/'
            html = await scraper.fetch(url)
            if not html:
                continue

            # Extract names from JSON-LD ItemList
            names: List[str] = []
            for block in LD_JSON_RE.findall(html):
                try:
                    data = json.loads(block)
                except (json.JSONDecodeError, ValueError):
                    continue
                if data.get('@type') == 'ItemList' and 'itemListElement' in data:
                    for item in data['itemListElement']:
                        biz = item.get('item', {})
                        name = biz.get('name', '')
                        if name:
                            names.append(name)

            # Extract phones from tel: links
            tel_phones_raw = TEL_HREF_RE.findall(html)
            tel_phones: List[str] = []
            for raw in tel_phones_raw:
                norm = normalize_ru_phone(raw.strip(), reject_non_ru=True)
                if norm and len(norm) == 12:
                    tel_phones.append(norm)

            # Match names ↔ phones by index (first tel: is often a duplicate/ad, rest match 1:1)
            if names and tel_phones:
                # If more phones than names, skip first (it's usually an ad/portal phone)
                offset = 1 if len(tel_phones) > len(names) else 0
                matched = min(len(names), len(tel_phones) - offset)
                for i in range(matched):
                    phone = tel_phones[offset + i]
                    # Skip fake/test numbers (consecutive digits, all same, etc.)
                    digits = phone[2:]  # strip +7
                    if digits == digits[0] * 10:  # e.g. 0000000000
                        continue
                    cat_inferred = infer_category(f'{cat} {names[i]}')
                    if scraper.add(LegitEntry(phone, names[i], cat_inferred, 'zoon', city_key, url)):
                        total += 1
                        city_added += 1
            # No fallback — regex phones from zoon are mostly ads/fake

            log.info(f"  zoon/{city_slug}/{cat}: names={len(names)}, phones={len(tel_phones)}, total={total}")
        # Persist after each city so a SIGTERM (per-iter timeout) doesn't
        # discard the partial work for the current source.
        _checkpoint(scraper)
        if city_added:
            log.info(f"  zoon city {city_key}: +{city_added} (total={total}) — checkpointed")
    return total


# ── Source 2: spravker.ru ─────────────────────────────────────────────────

# Per-category caps for the spravker deep-crawl. The previous values
# (only page 1, 10 org pages) were severely underutilising spravker —
# most categories paginate to 10-30 pages and have ~10-20 org cards per
# page (so the missed yield was ~5x). Bumping these to 8 listing pages
# and 30 org pages per category puts spravker in the same league as the
# OSM bulk shard while staying within per-iter wall-clock budgets.
SPRAVKER_MAX_LISTING_PAGES = 8
SPRAVKER_MAX_ORG_PAGES_PER_CATEGORY = 30


async def scrape_spravker_category(scraper: AsyncScraper, city_key: str,
                                    host: str, subcat: str) -> int:
    """Spravker: paginate subcategory listing → extract phones + .htm org links → visit each.

    The category listing paginates via ``?page=N`` (e.g. /bolnicy/?page=2).
    We discover the highest page number from pagination-list anchors on
    page 1 and then iterate through pages 1..min(max, found). Each page
    contributes both:
      * inline tel:/regex phones from the page itself, and
      * .htm org-page links which we visit individually.
    """
    total = 0

    # Step 1: listing page 1 — extract phones, org URLs, and page count.
    url = f'https://{host}/{subcat}/'
    html = await scraper.fetch(url)
    if not html:
        return 0

    def _harvest_listing(page_html: str, page_url: str) -> Tuple[int, List[str]]:
        """Return (added_listing_phones, .htm_org_links) for a listing page."""
        listing_phones: List[str] = []
        for raw in TEL_HREF_RE.findall(page_html):
            norm = normalize_ru_phone(raw.strip(), reject_non_ru=True)
            if norm and len(norm) == 12 and norm not in listing_phones:
                listing_phones.append(norm)
        if not listing_phones:
            listing_phones = extract_phones(page_html)
        added_inline = 0
        if listing_phones:
            title = extract_title(page_html)
            cat_inferred = infer_category(f'{subcat} {title}')
            added_inline = scraper.add_phones(
                listing_phones, title or subcat, cat_inferred,
                'spravker', city_key, page_url,
            )
        org_links_local: List[str] = []
        for link in SPRAVKER_ORG_RE.findall(page_html):
            if link.startswith('/'):
                full = f'https://{host}{link}'
            elif link.startswith('http'):
                full = link
            else:
                full = f'https://{host}/{subcat}/{link}'
            org_links_local.append(full)
        return added_inline, org_links_local

    added_inline, org_urls = _harvest_listing(html, url)
    total += added_inline

    # Detect highest pagination page from the listing's pagination-list
    discovered_max = _spravker_max_page(html)
    max_page = min(discovered_max, SPRAVKER_MAX_LISTING_PAGES)

    # Step 1b: paginate through 2..max_page if available
    for page_num in range(2, max_page + 1):
        page_url = f'https://{host}/{subcat}/?page={page_num}'
        page_html = await scraper.fetch(page_url)
        if not page_html:
            break  # gap → assume no more pages
        added_inline, more_org_urls = _harvest_listing(page_html, page_url)
        total += added_inline
        org_urls.extend(more_org_urls)

    # Dedup org URLs (different pages may surface the same org card)
    seen_org: Set[str] = set()
    unique_org_urls: List[str] = []
    for u in org_urls:
        if u not in seen_org:
            seen_org.add(u)
            unique_org_urls.append(u)

    log.info(
        f"  spravker_listing/{host}/{subcat}: pages={max_page}, "
        f"org_links={len(unique_org_urls)} (deduped from {len(org_urls)}), "
        f"inline_added={total}"
    )

    # Step 2: visit each org page — ONLY tel: links (regex on org pages
    # picks up too many spurious phone-shaped numbers from ads/footers).
    for org_url in unique_org_urls[:SPRAVKER_MAX_ORG_PAGES_PER_CATEGORY]:
        html = await scraper.fetch(org_url)
        if not html:
            continue
        tel_raw = TEL_HREF_RE.findall(html)
        org_phones: List[str] = []
        for raw in tel_raw:
            norm = normalize_ru_phone(raw.strip(), reject_non_ru=True)
            if norm and len(norm) == 12:
                org_phones.append(norm)
        if org_phones:
            title = extract_title(html)
            cat_inferred = infer_category(f'{subcat} {title}')
            added = scraper.add_phones(org_phones, title or subcat, cat_inferred, 'spravker', city_key, org_url)
            total += added

    return total


async def scrape_spravker(scraper: AsyncScraper, cities: Dict[str, str],
                           subcategories: List[str]) -> int:
    total = 0
    for city_key, host in cities.items():
        city_added = 0
        for subcat in subcategories:
            count = await scrape_spravker_category(scraper, city_key, host, subcat)
            total += count
            city_added += count
            if count > 0:
                log.info(f"  spravker/{city_key}/{subcat}: {count}")
        # Same rationale as zoon: 51-city fan-out may be SIGTERM'd by the
        # per-iter timeout, so persist progress per city.
        _checkpoint(scraper)
        if city_added:
            log.info(f"  spravker city {city_key}: +{city_added} (total={total}) — checkpointed")
    return total


# ── Source 3: rusprofile.ru ────────────────────────────────────────────────

async def scrape_rusprofile(scraper: AsyncScraper, queries: List[str]) -> int:
    """Rusprofile: search → collect /id/XXXX links → visit company pages for tel: links."""
    total = 0
    for query in queries:
        company_ids: Set[str] = set()

        # Step 1: collect company IDs from search results (2 pages)
        for page in range(1, 3):
            url = f'https://www.rusprofile.ru/search?query={query}&page={page}'
            html = await scraper.fetch(url, allow_status={200, 403})
            if not html:
                break
            ids = RUSPROFILE_ORG_RE.findall(html)
            company_ids.update(ids)
            if not ids:
                break

        if not company_ids:
            continue

        log.info(f"  rusprofile/{query}: {len(company_ids)} companies found")

        # Step 2: visit each company page for phone
        for cid in list(company_ids)[:5]:
            url = f'https://www.rusprofile.ru{cid}'
            html = await scraper.fetch(url, allow_status={200, 403})
            if not html:
                continue

            # Focus on tel: links (most reliable on rusprofile)
            tel_phones: List[str] = []
            for raw in TEL_HREF_RE.findall(html):
                norm = normalize_ru_phone(raw.strip(), reject_non_ru=True)
                if norm and len(norm) == 12:
                    tel_phones.append(norm)

            if tel_phones:
                title = extract_title(html)
                cat_inferred = infer_category(f'{query} {title}')
                added = scraper.add_phones(tel_phones, title or query, cat_inferred, 'rusprofile', '', url)
                total += added
                if added > 0:
                    log.info(f"  rusprofile_org {cid}: {added} new (total {total})")

    return total


# ── Source 4: mosgorzdrav.ru ───────────────────────────────────────────────

async def scrape_mosgorzdrav(scraper: AsyncScraper) -> int:
    total = 0
    for page_url in [
        'https://mosgorzdrav.ru/',
        'https://mosgorzdrav.ru/ru/healthcare/',
        'https://mosgorzdrav.ru/ru/contacts/',
    ]:
        html = await scraper.fetch(page_url)
        if not html:
            continue
        phones = extract_phones(html)
        if phones:
            added = scraper.add_phones(phones, 'Мосгорздрав', 'medical', 'mosgorzdrav', 'msk', page_url)
            total += added
    return total


# ── Source 5: mos.ru ──────────────────────────────────────────────────────

async def scrape_mos_ru(scraper: AsyncScraper) -> int:
    total = 0
    for page_url in [
        'https://www.mos.ru/',
        'https://www.mos.ru/contacts/',
        'https://www.mos.ru/government/contacts/',
    ]:
        html = await scraper.fetch(page_url)
        if not html:
            continue
        phones = extract_phones(html)
        if phones:
            added = scraper.add_phones(phones, 'mos.ru', 'government', 'mos_ru', 'msk', page_url)
            total += added
    return total


# ── Source 6: cian.ru — real estate agents/owners ──────────────────────────

async def scrape_cian(scraper: AsyncScraper, max_pages: int = 30) -> int:
    """ЦИАН — телефоны агентов и собственников недвижимости."""
    total = 0
    regions = [1, 2, 4593, 4597, 4601, 4605, 4609, 4612, 4615, 4618]  # МСК, СПБ, и др.
    deal_types = ['sale', 'rent']
    offer_types = ['flat', 'house', 'commercial']

    for region in regions:
        for deal in deal_types:
            for offer in offer_types:
                for page in range(1, max_pages + 1):
                    url = f'https://www.cian.ru/cat.php?deal_type={deal}&engine_version=2&offer_type={offer}&region={region}&p={page}'
                    html = await scraper.fetch(url)
                    if not html:
                        break
                    phones = extract_phones(html)
                    if not phones and page > 3:
                        break
                    if phones:
                        cat = 'realestate_agent' if 'homeowner' not in url else 'realestate_owner'
                        added = scraper.add_phones(phones, 'ЦИАН', cat, 'cian', '', url)
                        total += added
                        if added > 0:
                            log.info(f"  cian/{deal}/{offer}/r{region}/p{page}: {added} new (total {total})")

    # Owner-only listings
    for region in regions:
        for page in range(1, min(max_pages, 10) + 1):
            url = f'https://www.cian.ru/cat.php?deal_type=rent&engine_version=2&offer_type=flat&region={region}&is_by_homeowner=1&p={page}'
            html = await scraper.fetch(url)
            if not html:
                break
            phones = extract_phones(html)
            if not phones and page > 2:
                break
            if phones:
                added = scraper.add_phones(phones, 'ЦИАН собственник', 'realestate_owner', 'cian', '', url)
                total += added

    return total


async def scrape_public_url_list(scraper: AsyncScraper, source: str,
                                 urls: List[Tuple[str, str, str]],
                                 link_limit: int = 8,
                                 confidence: float = 0.70,
                                 allow_status: Optional[Set[int]] = None) -> int:
    """Walk a static URL list, extract phones from each page + top N same-host links.

    ``allow_status`` is forwarded to ``scraper.fetch`` and lets callers accept
    pages that the origin returns with non-200 codes (e.g. soft-404s with real
    content — common for Russian press/insurance contact pages where the site
    returns 404 status alongside a fully-rendered contacts block).
    """
    total = 0
    for page_url, name, category in urls:
        html = await scraper.fetch(page_url, allow_status=allow_status)
        if not html:
            continue
        title = extract_title(html)
        phones = extract_phones(html)
        if phones:
            added = scraper.add_phones(phones, title or name, category, source, '', page_url, confidence)
            total += added
            if added > 0:
                log.info(f"  {source}/{category}: {added} new from {page_url}")
        for link in extract_links(html, page_url, limit=link_limit):
            html2 = await scraper.fetch(link, allow_status=allow_status)
            if not html2:
                continue
            phones2 = extract_phones(html2)
            if phones2:
                title2 = extract_title(html2)
                added = scraper.add_phones(phones2, title2 or title or name, category, source, '', link, confidence)
                total += added
                if added > 0:
                    log.info(f"  {source}/{category}: {added} new from detail")
    return total


async def scrape_delivery_public(scraper: AsyncScraper) -> int:
    return await scrape_public_url_list(scraper, 'delivery_public', DELIVERY_PUBLIC_URLS, link_limit=5, confidence=0.85)


async def scrape_service_marketplaces(scraper: AsyncScraper) -> int:
    urls = list(SERVICE_MARKETPLACE_URLS)
    for query in FREELANCE_QUERIES:
        urls.append((f'https://freelance.ru/search/?q={quote(query)}', f'Freelance.ru {query}', 'freelancer'))
        urls.append((f'https://www.fl.ru/search/?type=users&search_string={quote(query)}', f'FL.ru {query}', 'freelancer'))
    return await scrape_public_url_list(scraper, 'service_marketplace', urls, link_limit=10, confidence=0.60)


async def scrape_service_marketplaces_fast(scraper: AsyncScraper) -> int:
    urls = list(FAST_SERVICE_MARKETPLACE_URLS)
    for query in FREELANCE_QUERIES:
        urls.append((f'https://freelance.ru/search/?q={quote(query)}', f'Freelance.ru {query}', 'freelancer'))
        urls.append((f'https://www.fl.ru/search/?type=users&search_string={quote(query)}', f'FL.ru {query}', 'freelancer'))
    return await scrape_public_url_list(scraper, 'service_marketplace_fast', urls, link_limit=6, confidence=0.60)


async def scrape_classified_public(scraper: AsyncScraper) -> int:
    return await scrape_public_url_list(scraper, 'classified_public', CLASSIFIED_PUBLIC_URLS, link_limit=12, confidence=0.55)


# ── Источники: крупные РФ-организации с публичными контактными страницами ──
#
# Все они выкачиваются через общий ``scrape_public_url_list``: фетчим главную
# страницу контактов, добавляем телефоны напрямую, плюс по верхним ссылкам
# того же домена идём на вложенные страницы (региональные офисы / филиалы).
# ``confidence=0.85`` — стандартный уровень для горячих линий и контактов
# крупных компаний (банк/телеком/ритейл/госорганы); 0.95 для федеральных
# горячих линий (то же что у official_hotline).

# Ряд РФ-сайтов отдаёт статус 404 при полностью валидном HTML-контенте (особенно
# банки/страховые/СМИ с anti-bot WAF). Принимаем их явно — номера всё равно
# валидируются общим is_plausible_phone (формат РФ, def-коды, фильтр плейсхолдеров).
LENIENT_STATUS = {200, 404, 410}


async def scrape_ru_banks(scraper: AsyncScraper) -> int:
    """Контактные страницы крупных РФ-банков (горячие линии и офисы)."""
    return await scrape_public_url_list(
        scraper, 'ru_banks', RU_BANK_CONTACT_URLS, link_limit=8,
        confidence=0.85, allow_status=LENIENT_STATUS
    )


async def scrape_ru_telecom(scraper: AsyncScraper) -> int:
    """Контакты МегаФон/МТС/Билайн/Tele2 — горячие линии, региональные центры."""
    return await scrape_public_url_list(
        scraper, 'ru_telecom', RU_TELECOM_CONTACT_URLS, link_limit=6,
        confidence=0.85, allow_status=LENIENT_STATUS
    )


async def scrape_ru_airlines(scraper: AsyncScraper) -> int:
    """Контактные страницы РФ-авиакомпаний — call-центры и представительства."""
    return await scrape_public_url_list(
        scraper, 'ru_airlines', RU_AIRLINE_CONTACT_URLS, link_limit=6,
        confidence=0.85, allow_status=LENIENT_STATUS
    )


async def scrape_ru_airports(scraper: AsyncScraper) -> int:
    """Контактные страницы крупных РФ-аэропортов (справочные службы, терминалы)."""
    return await scrape_public_url_list(
        scraper, 'ru_airports', RU_AIRPORT_CONTACT_URLS, link_limit=8,
        confidence=0.85, allow_status=LENIENT_STATUS
    )


async def scrape_ru_press(scraper: AsyncScraper) -> int:
    """Контакты редакций крупных СМИ (АиФ, КП, Коммерсантъ, Ведомости и др.)."""
    return await scrape_public_url_list(
        scraper, 'ru_press', RU_PRESS_CONTACT_URLS, link_limit=4,
        confidence=0.80, allow_status=LENIENT_STATUS
    )


async def scrape_ru_universities(scraper: AsyncScraper) -> int:
    """Контактные страницы крупных РФ-вузов (приёмные комиссии, деканаты)."""
    return await scrape_public_url_list(
        scraper, 'ru_universities', RU_UNIVERSITY_CONTACT_URLS, link_limit=6,
        confidence=0.85, allow_status=LENIENT_STATUS
    )


async def scrape_ru_insurance(scraper: AsyncScraper) -> int:
    """Контакты страховых компаний (РГС, СОГАЗ, Ингосстрах, ВСК и др.)."""
    return await scrape_public_url_list(
        scraper, 'ru_insurance', RU_INSURANCE_CONTACT_URLS, link_limit=6,
        confidence=0.85, allow_status=LENIENT_STATUS
    )


async def scrape_ru_retail(scraper: AsyncScraper) -> int:
    """Контакты крупных розничных сетей (Магнит, ВкусВилл, Metro, Дикси и др.)."""
    return await scrape_public_url_list(
        scraper, 'ru_retail', RU_RETAIL_CONTACT_URLS, link_limit=6,
        confidence=0.85, allow_status=LENIENT_STATUS
    )


async def scrape_ru_federal_hotlines(scraper: AsyncScraper) -> int:
    """Контакты федеральных ведомств: ФНС, СФР, МЧС, Роскомнадзор, ФССП и др."""
    # link_limit=2 чтобы избежать долгих вызовов на тяжёлых гос-порталах (sfr.gov.ru и пр.
    # часто отвечают медленно и обычно все нужные номера уже есть на самой странице контактов).
    return await scrape_public_url_list(
        scraper, 'ru_federal_hotlines', RU_FEDERAL_HOTLINE_URLS, link_limit=2,
        confidence=0.95, allow_status=LENIENT_STATUS
    )


async def scrape_ru_marketplaces(scraper: AsyncScraper) -> int:
    """Контакты поддержки маркетплейсов (Яндекс Маркет, Mail.ru, Lamoda)."""
    return await scrape_public_url_list(
        scraper, 'ru_marketplaces', RU_MARKETPLACE_CONTACT_URLS, link_limit=4,
        confidence=0.80, allow_status=LENIENT_STATUS
    )


# ── Source 7: fl.ru — freelancers with public contacts ────────────────────

async def scrape_fl_ru(scraper: AsyncScraper) -> int:
    """fl.ru — фрилансеры с публичными контактами."""
    total = 0
    for page_url in [
        'https://www.fl.ru/freelancers/',
        'https://www.fl.ru/projects/',
    ]:
        html = await scraper.fetch(page_url)
        if not html:
            continue
        phones = extract_phones(html)
        if phones:
            added = scraper.add_phones(phones, 'FL.ru', 'freelancer', 'fl_ru', '', page_url)
            total += added
    return total


# ── Source 8: hands.ru — freelancers / masters ────────────────────────────

async def scrape_hands_ru(scraper: AsyncScraper) -> int:
    """hands.ru — фрилансеры и мастера с публичными контактами."""
    total = 0
    for page_url in [
        'https://hands.ru/',
        'https://hands.ru/programmers/',
        'https://hands.ru/designers/',
        'https://hands.ru/translators/',
        'https://hands.ru/copywriters/',
    ]:
        html = await scraper.fetch(page_url)
        if not html:
            continue
        phones = extract_phones(html)
        if phones:
            added = scraper.add_phones(phones, 'Hands.ru', 'freelancer', 'hands_ru', '', page_url)
            total += added
    return total


# ── Source 9: freelance.ru — freelancers ──────────────────────────────────

async def scrape_freelance_ru(scraper: AsyncScraper) -> int:
    """freelance.ru — фрилансеры с публичными контактами."""
    total = 0
    for page_url in [
        'https://freelance.ru/',
        'https://freelance.ru/projects/',
        'https://freelance.ru/search/?q=разработчик',
        'https://freelance.ru/search/?q=дизайнер',
    ]:
        html = await scraper.fetch(page_url)
        if not html:
            continue
        phones = extract_phones(html)
        if phones:
            added = scraper.add_phones(phones, 'Freelance.ru', 'freelancer', 'freelance_ru', '', page_url)
            total += added
    return total


# ── Source 10: drom.ru — car sellers / baza ──────────────────────────────

async def scrape_drom(scraper: AsyncScraper) -> int:
    """drom.ru / baza.drom.ru — частные продавцы авто."""
    total = 0
    for page_url in [
        'https://baza.drom.ru/',
        'https://auto.drom.ru/region77/',
        'https://auto.drom.ru/moskva/',
        'https://auto.drom.ru/sankt-peterburg/',
    ]:
        html = await scraper.fetch(page_url)
        if not html:
            continue
        phones = extract_phones(html)
        if phones:
            added = scraper.add_phones(phones, 'Drom.ru', 'private_seller', 'drom', '', page_url)
            total += added
    return total


# ── Source 11: orgpage.ru — RU business directory (millions of pages) ─────

# Homepage at https://www.orgpage.ru/ exposes a public list of "kompanii-X-Y"
# range-pages that enumerate companies by ID (~60k companies per range, dozens
# of ranges, total ~2-4M companies). Each company page is at
# /<city>/<slug>-<id>.html and typically shows ~20 phone numbers (HQ +
# branches). Confirmed working from this VM with no anti-bot.
# Two-level URL hierarchy:
#   /                         → ~51 top-level kompanii-A-B/ links (each
#                               spans ~64k IDs)
#   /kompanii-A-B/            → ~250 sub-range kompanii-X-Y/ links (each
#                               spans ~258 IDs)
#   /kompanii-X-Y/            → ~259 actual company .html pages
#                               (e.g. /moskva/some-org-12345.html)
#   /<city>/<slug>-<id>.html  → company detail page with phones in tel:
#                               links and structured contact blocks
ORGPAGE_RANGE_RE = re.compile(r'href="(?:https?://[^"]+)?(/kompanii-\d+-\d+/)"')
ORGPAGE_COMPANY_RE = re.compile(r'href="(?:https?://[^"]+)?(/[a-z0-9_-]+/[a-z0-9_-]+-\d+\.html)"', re.I)

async def scrape_orgpage(scraper: AsyncScraper, max_ranges: int = 6,
                         max_companies_per_range: int = 25) -> int:
    """orgpage.ru: 2-level walk top-range → sub-range → company → phones.

    The catalog is sharded as a tree: 51 top-level ranges, each containing
    ~250 sub-ranges, each containing ~259 company pages. We pick
    ``max_ranges`` random sub-ranges (across random top-ranges) and grab
    ``max_companies_per_range`` companies from each. Random sampling means
    successive iterations naturally explore disjoint slices of the ~3.4M
    companies; resume-state handles full-cycle dedup.
    """
    total = 0
    home = await scraper.fetch('https://www.orgpage.ru/')
    if not home:
        return 0
    top_ranges = list(set(ORGPAGE_RANGE_RE.findall(home)))
    if not top_ranges:
        return 0

    # Pick a few top-ranges at random, then for each grab some sub-ranges.
    top_pick = _shuffle_sample(top_ranges, max(2, max_ranges // 3))

    sub_ranges: List[str] = []
    for tr in top_pick:
        url = f'https://www.orgpage.ru{tr}'
        html = await scraper.fetch(url)
        if not html:
            continue
        subs = list(set(ORGPAGE_RANGE_RE.findall(html)))
        # Filter out the top-range itself
        subs = [s for s in subs if s != tr]
        sub_ranges.extend(subs)

    if not sub_ranges:
        return 0

    sub_ranges = _shuffle_sample(sub_ranges, max_ranges)

    for sr in sub_ranges:
        range_added = 0
        url = f'https://www.orgpage.ru{sr}'
        html = await scraper.fetch(url)
        if not html:
            continue
        companies = list(set(ORGPAGE_COMPANY_RE.findall(html)))
        # Skip non-company pages like /about.html, /privacy.html
        companies = [c for c in companies if not c.startswith(
            ('/about', '/privacy', '/terms', '/landing', '/contact', '/help'))]
        companies = _shuffle_sample(companies, max_companies_per_range)
        for c in companies:
            curl = f'https://www.orgpage.ru{c}'
            page = await scraper.fetch(curl)
            if not page:
                continue
            phones = extract_phones(page)
            if not phones:
                continue
            title = extract_title(page) or 'Org'
            cat_inferred = infer_category(title)
            added = scraper.add_phones(phones, title, cat_inferred or 'business',
                                        'orgpage', '', curl)
            total += added
            range_added += added
            if added:
                _checkpoint(scraper)
        if range_added:
            log.info(f"  orgpage/{sr}: +{range_added} (total={total}) — checkpointed")
    return total


# ── Source: list-org.com (RU legal-entity directory) ──────────────────────
#
# list-org.com publishes a sitemap-index plus per-region sitemaps. Each
# company page exposes phone numbers in a `tel:` link or in the body text
# adjacent to «Телефон». We walk the sitemap index → sub-sitemap → company
# page chain and pick a random subset each iteration so successive crawl
# loops cover disjoint slices of the ~3-4M-entry catalog.

LIST_ORG_SITEMAP_INDEX = 'https://www.list-org.com/sitemap.xml'
LIST_ORG_SITEMAP_RE = re.compile(r'<loc>\s*(https?://[^<\s]+)\s*</loc>', re.I)
LIST_ORG_COMPANY_RE = re.compile(r'<loc>\s*(https?://(?:www\.)?list-org\.com/company/\d+[^<\s]*)\s*</loc>', re.I)


async def scrape_list_org(scraper: AsyncScraper,
                          max_subsitemaps: int = 4,
                          max_companies: int = 250) -> int:
    """Walk list-org.com sitemap index → random sub-sitemaps → company pages."""
    total = 0
    index_html = await scraper.fetch(LIST_ORG_SITEMAP_INDEX)
    if not index_html:
        return 0
    sub_urls = [
        u for u in LIST_ORG_SITEMAP_RE.findall(index_html)
        if 'list-org.com' in u and u.endswith('.xml')
    ]
    if not sub_urls:
        return 0

    sub_urls = _shuffle_sample(sub_urls, max_subsitemaps)

    company_urls: List[str] = []
    for sub_url in sub_urls:
        sub_html = await scraper.fetch(sub_url)
        if not sub_html:
            continue
        for cu in LIST_ORG_COMPANY_RE.findall(sub_html):
            company_urls.append(cu.split('#', 1)[0])

    if not company_urls:
        return 0

    company_urls = _shuffle_sample(company_urls, max_companies)

    checkpoint_added = 0
    for curl in company_urls:
        page = await scraper.fetch(curl)
        if not page:
            continue
        phones = extract_phones(page)
        if not phones:
            continue
        title = extract_title(page) or 'Организация (list-org)'
        cat = infer_category(title) or 'business'
        added = scraper.add_phones(phones, title[:200], cat,
                                    'list_org', '', curl, source_confidence=0.78)
        total += added
        checkpoint_added += added
        if checkpoint_added >= 25:
            _checkpoint(scraper)
            log.info(f"  list_org: +{checkpoint_added} since checkpoint (total={total})")
            checkpoint_added = 0
    if checkpoint_added:
        _checkpoint(scraper)
        log.info(f"  list_org: +{checkpoint_added} since checkpoint (total={total})")
    return total


# ── Source: 2GIS Catalog API ──────────────────────────────────────────────
#
# Public catalog REST API at catalog.api.2gis.com. Without an API key we
# silently skip — registering one is free at https://dev.2gis.ru . With a
# key the free-tier yields ~50k requests/day, easily +30-50k ALLOW
# numbers/day given the average 1-3 phones per business item.

DGIS_API_BASE = 'https://catalog.api.2gis.com/3.0'
# Region IDs from 2GIS' public regions endpoint. Top-15 cities by item volume.
DGIS_REGIONS = [
    1,    # Москва
    2,    # Санкт-Петербург
    32,   # Новосибирск
    38,   # Екатеринбург
    52,   # Казань
    49,   # Нижний Новгород
    20,   # Ростов-на-Дону
    66,   # Уфа
    8,    # Краснодар
    11,   # Воронеж
    78,   # Челябинск
    18,   # Самара
    61,   # Пермь
    23,   # Волгоград
    74,   # Тюмень
]
DGIS_QUERIES = [
    'кафе', 'ресторан', 'столовая', 'пиццерия', 'кофейня',
    'аптека', 'клиника', 'стоматология', 'медицинский центр',
    'банк', 'страховая', 'нотариус', 'адвокат',
    'автосервис', 'автомойка', 'шиномонтаж', 'автосалон',
    'парикмахерская', 'салон красоты', 'фитнес',
    'магазин продуктов', 'супермаркет', 'товары для дома',
    'детский сад', 'школа', 'университет',
    'гостиница', 'хостел', 'туристическое агентство',
    'доставка еды', 'такси', 'грузоперевозки',
    'ремонт квартир', 'окна', 'мебель',
    'интернет провайдер', 'управляющая компания',
    'ветеринарная клиника', 'зоомагазин',
]


async def scrape_2gis_api(scraper: AsyncScraper,
                          api_key: str = '',
                          max_regions: int = 6,
                          max_queries_per_region: int = 6,
                          max_pages_per_query: int = 5) -> int:
    """Query 2GIS catalog API and harvest contact phones from item.contact_groups."""
    api_key = api_key or TWO_GIS_API_KEY
    if not api_key:
        log.info("  2GIS API key not set (TWO_GIS_API_KEY); skipping")
        return 0

    total = 0
    region_pool = list(DGIS_REGIONS)
    random.shuffle(region_pool)
    region_pool = region_pool[:max_regions]

    for region_id in region_pool:
        query_pool = list(DGIS_QUERIES)
        random.shuffle(query_pool)
        for query in query_pool[:max_queries_per_region]:
            for page in range(1, max_pages_per_query + 1):
                params = (
                    f'q={quote(query)}&region_id={region_id}'
                    f'&fields=items.contact_groups,items.name,items.address_name'
                    f'&page={page}&page_size=50&key={quote(api_key)}'
                )
                url = f'{DGIS_API_BASE}/items?{params}'
                body = await scraper.fetch(url)
                if not body:
                    break
                try:
                    payload = json.loads(body)
                except (ValueError, TypeError):
                    break
                items = ((payload or {}).get('result') or {}).get('items') or []
                if not items:
                    break
                for item in items:
                    name = (item.get('name') or '').strip() or 'Организация (2gis)'
                    address = (item.get('address_name') or '').strip()
                    title = f'{name}'
                    if address:
                        title = f'{name} — {address}'
                    phones: List[str] = []
                    for group in item.get('contact_groups') or []:
                        for contact in group.get('contacts') or []:
                            if (contact.get('type') or '').lower() != 'phone':
                                continue
                            raw = contact.get('value') or contact.get('text') or ''
                            norm = normalize_ru_phone(str(raw), reject_non_ru=True)
                            if is_plausible_phone(norm) and norm not in phones:
                                phones.append(norm)
                    if not phones:
                        continue
                    cat = infer_category(title) or 'business'
                    added = scraper.add_phones(
                        phones, title[:200], cat, '2gis', '',
                        f'https://2gis.ru/firm/{item.get("id", "")}',
                        source_confidence=0.88,
                    )
                    total += added
                if len(items) < 50:
                    break
    return total


# ── Bonus: official numbers ────────────────────────────────────────────────

def load_official_whitelist() -> List[LegitEntry]:
    entries = []
    csv_path = os.path.normpath(os.path.join(
        os.path.dirname(__file__), '..', 'datasets', 'ru', 'raw', 'whitelist_official_ru.csv'
    ))
    if not os.path.exists(csv_path):
        return entries
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            num = row.get('normalized_number', '').strip()
            if num:
                entries.append(LegitEntry(
                    normalized_number=num,
                    name=row.get('name', ''),
                    category=row.get('category', 'other'),
                    source='official_whitelist',
                    city='',
                    url='',
                    source_confidence=0.95
                ))
    return entries


_RAW_OFFICIAL_HOTLINES: List[Tuple[str, str, str]] = [
    # ── Delivery / e-commerce / takeaway ──
    ('+78005503355', 'Яндекс Еда', 'delivery'),
    ('+78005551333', 'Delivery Club', 'delivery'),
    ('+78005004003', 'Самокат', 'delivery'),
    ('+78005553300', 'СберМаркет', 'delivery'),
    ('+78002000600', 'Озон', 'delivery'),
    ('+78007007777', 'Wildberries', 'delivery'),
    ('+78005551777', 'Яндекс Маркет', 'delivery'),
    ('+78007003388', 'DPD', 'delivery'),
    ('+78001003900', 'Boxberry', 'delivery'),
    ('+78007002000', 'CDEK', 'delivery'),
    ('+78007001000', 'ПЭК', 'delivery'),
    ('+78007007575', 'Почта России', 'delivery'),
    ('+78002340000', 'Достависта', 'delivery'),
    ('+78007005588', 'Яндекс Лавка', 'delivery'),
    ('+78002001880', 'Яндекс Доставка', 'delivery'),

    # ── Insurance ──
    ('+78002000900', 'Росгосстрах', 'insurance'),
    ('+78003330999', 'АльфаСтрахование', 'insurance'),
    ('+78002341802', 'РЕСО-Гарантия', 'insurance'),
    ('+78007550001', 'Согласие', 'insurance'),
    ('+78001007755', 'Ингосстрах', 'insurance'),
    ('+78005550555', 'ВСК (Военно-страховая)', 'insurance'),
    ('+78007007707', 'Согаз', 'insurance'),
    ('+78007757755', 'МАКС', 'insurance'),
    ('+78007003738', 'Сбер Страхование', 'insurance'),

    # ── Retail / FMCG ──
    ('+78005555505', 'Пятёрочка', 'retail'),
    ('+78002009002', 'Магнит', 'retail'),
    ('+78007004111', 'Лента', 'retail'),
    ('+78002009555', 'Перекрёсток', 'retail'),
    ('+78002009595', 'Ашан', 'retail'),
    ('+78002005565', 'Метро Cash & Carry', 'retail'),
    ('+78002007575', 'Вкусвилл', 'retail'),
    ('+78007005757', 'Детский Мир', 'retail'),
    ('+78007077777', 'М.Видео', 'retail'),
    ('+78005553311', 'Эльдорадо', 'retail'),
    ('+78007770201', 'DNS', 'retail'),
    ('+78007007474', 'Ситилинк', 'retail'),
    ('+78007075578', 'Спортмастер', 'retail'),

    # ── Government / federal hotlines (high trust) ──
    ('+78001007010', 'Госуслуги', 'government'),
    ('+74957737200', 'Госуслуги Москва', 'government'),
    ('+78005554904', 'Роспотребнадзор', 'government'),
    ('+78006000443', 'Пенсионный фонд', 'government'),
    ('+78001000001', 'СФР Социальный фонд', 'government'),
    ('+78005500550', 'ФНС', 'government'),
    ('+78005500222', 'ФСС', 'government'),
    ('+78002227777', 'ЦБ РФ', 'government'),
    ('+74957719100', 'ЦБ РФ Москва', 'government'),
    ('+78002224747', 'Росреестр', 'government'),
    ('+78002001112', 'МВД России', 'government'),
    ('+78007770114', 'МЧС Россия', 'government'),
    ('+74953836900', 'МЧС Москва', 'government'),
    ('+78007007171', 'СКР Следком', 'government'),
    ('+78008008555', 'ГенПрокуратура', 'government'),
    ('+78005552323', 'Роструд', 'government'),
    ('+78005556700', 'Минцифры', 'government'),
    ('+78005000035', 'Правительство РФ', 'government'),
    ('+78001003510', 'Минтруд', 'government'),

    # ── Top banks (CB RF licensed, public hotlines) ──
    ('+78005553535', 'Сбербанк', 'bank'),
    ('+74955005550', 'Сбербанк Москва', 'bank'),
    ('+78007007535', 'Райффайзенбанк', 'bank'),
    ('+78002000023', 'ВТБ', 'bank'),
    ('+78002005959', 'T-Банк (Тинькофф)', 'bank'),
    ('+78002000000', 'Альфа-Банк', 'bank'),
    ('+78007007007', 'Газпромбанк', 'bank'),
    ('+78002007777', 'Россельхозбанк', 'bank'),
    ('+78004445555', 'Открытие', 'bank'),
    ('+78007007666', 'МКБ Московский Кредитный', 'bank'),
    ('+78001000404', 'Уралсиб', 'bank'),
    ('+78002000202', 'АК Барс', 'bank'),
    ('+78002000125', 'Банк Россия', 'bank'),
    ('+78007002424', 'Почта Банк', 'bank'),
    ('+78007003500', 'Дом.РФ (ДОМ.РФ)', 'bank'),
    ('+78001008383', 'ОТП-Банк', 'bank'),
    ('+78005553737', 'Хоум Банк', 'bank'),
    ('+78007007711', 'Ренессанс Кредит', 'bank'),
    ('+78007007373', 'Банк С.Петербург', 'bank'),
    ('+78002001525', 'СМП Банк', 'bank'),
    ('+78007005577', 'ЮНИКРЕДИТ Банк', 'bank'),
    ('+78002007799', 'Абсолют Банк', 'bank'),

    # ── Healthcare hotlines ──
    ('+78002000200', 'Минздрав РФ', 'medical'),
    ('+74957776767', 'Депздрав Москвы', 'medical'),
    ('+78002000122', 'Детский телефон доверия', 'medical'),
    ('+78002007373', 'МедСи', 'medical'),
    ('+78002009909', 'INVITRO', 'medical'),
    ('+78002000404', 'Гемотест', 'medical'),
    ('+78002007010', 'KDL', 'medical'),
    ('+74953632424', 'ЕМС', 'medical'),

    # ── Transport / aviation / railway ──
    ('+78007750000', 'РЖД', 'transport'),
    ('+74956056555', 'РЖД Москва', 'transport'),
    ('+78002005577', 'S7 Airlines', 'transport'),
    ('+78007003007', 'Утаир', 'transport'),
    ('+78007551155', 'Россия Авиалинии', 'transport'),
    ('+78007077717', 'Победа', 'transport'),
    ('+78001001212', 'Яндекс Такси', 'transport'),
    ('+78007003434', 'Citymobil', 'transport'),
    ('+78007075050', 'Maxim Такси', 'transport'),
    ('+78001000050', 'Gett', 'transport'),
    ('+78007075080', 'Аэроэкспресс', 'transport'),
    ('+74956567000', 'Мосгортранс', 'transport'),
    ('+74953395800', 'Московский метрополитен', 'transport'),

    # ── Telecom (additional) ──
    ('+78003330890', 'МТС', 'telecom'),
    ('+78007000611', 'Билайн', 'telecom'),
    ('+78005500500', 'МегаФон', 'telecom'),
    ('+78005550607', 'Tele2', 'telecom'),
    ('+78001000800', 'Ростелеком', 'telecom'),
    ('+78001005050', 'Дом.ru', 'telecom'),
    ('+78007770800', 'АКАДО', 'telecom'),
    ('+78007075009', 'ТТК', 'telecom'),
    ('+78007750725', 'Триколор ТВ', 'telecom'),

    # ── Utilities (ЖКХ, энерго) ──
    ('+74959811919', 'МОСЭНЕРГОСБЫТ', 'utility'),
    ('+78007550555', 'Россети', 'utility'),
    ('+74955395353', 'Мосводоканал', 'utility'),
]


def _dedupe_hotlines(rows: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """First entry wins per phone number; later category collisions are dropped.

    Hardcoded hotline lists tend to grow over time and pick up duplicates as
    organisations share corporate-wide 8800 lines (e.g. multiple subsidiaries).
    The runtime ``scraper.add()`` already dedupes by ``normalized_number``,
    but we sanitise here so the *constant* list is also clean and so the
    "Added N hardcoded hotlines" log line reflects reality.
    """
    seen: Set[str] = set()
    out: List[Tuple[str, str, str]] = []
    for num, name, cat in rows:
        if num in seen:
            continue
        seen.add(num)
        out.append((num, name, cat))
    return out


OFFICIAL_HOTLINES: List[Tuple[str, str, str]] = _dedupe_hotlines(_RAW_OFFICIAL_HOTLINES)


# ── Source 6: synthetic user numbers from numbering plan ─────────────────

def generate_user_numbers(scraper: AsyncScraper, count: int = 5000) -> int:
    """Generate weak background mobile numbers from the official numbering plan.
    
    Uses DEF/ABC ranges from ru_numbering_plan.csv to create numbers that
    are guaranteed to be valid (existing operator + region combinations).
    Cross-checked against blacklist to exclude known fraud numbers. These rows are
    intentionally low-confidence and should not be treated like verified org phones.
    """
    plan_path = os.path.normpath(os.path.join(
        os.path.dirname(__file__), '..', 'datasets', 'ru', 'raw', 'ru_numbering_plan.csv'
    ))
    if not os.path.exists(plan_path):
        log.warning(f"  Numbering plan not found at {plan_path}, run ru_numbering_plan.py first")
        return 0

    # Load ranges — mobile only (16K ranges, fast)
    ranges = []
    with open(plan_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            ntype = row.get('number_type', '')
            if ntype != 'mobile':
                continue
            def_code = row.get('def_code', '').strip()
            start = int(row.get('start_number', '0'))
            end = int(row.get('end_number', '0'))
            operator = row.get('operator', '').strip()
            region = row.get('region', '').strip()
            capacity = end - start + 1
            if capacity > 0 and def_code:
                ranges.append({
                    'def_code': def_code,
                    'start': start, 'end': end,
                    'operator': operator, 'region': region,
                    'type': ntype, 'capacity': capacity,
                })

    if not ranges:
        log.warning("  No mobile ranges loaded from numbering plan")
        return 0

    log.info(f"  Loaded {len(ranges)} mobile ranges from numbering plan")

    # Pre-compute cumulative distribution for fast sampling
    import bisect
    cum_weights = []
    total = 0
    for r in ranges:
        total += r['capacity']
        cum_weights.append(total)

    # Generate numbers
    added = 0
    attempts = 0
    max_attempts = count * 3

    while added < count and attempts < max_attempts:
        attempts += 1
        # Pick a range using binary search on cumulative weights
        pick = random.randint(1, total)
        idx = bisect.bisect_left(cum_weights, pick)
        r = ranges[idx]
        # Pick a random subscriber number within the range
        subscriber = random.randint(r['start'], r['end'])
        # Build full number
        number = f'+7{r["def_code"]}{subscriber:07d}'

        # Validate
        if len(number) != 12:
            continue
        if number in scraper.seen:
            continue
        if number in scraper.blacklist:
            continue

        # Category and name from operator/region
        cat = 'personal_mobile'
        name = f'{r["operator"]}, {r["region"]}' if r['operator'] else r['region']

        if scraper.add(LegitEntry(number, name, cat, 'numbering_plan_background', r['region'], '', 0.25)):
            added += 1

    log.info(f"  Generated {added} user numbers ({attempts} attempts, {len(scraper.blacklist)} blacklist filtered)")
    return added


# ── Main ───────────────────────────────────────────────────────────────────

def build_source_registry(scraper: AsyncScraper,
                          spravker_cities: Dict[str, str],
                          zoon_cities: Dict[str, str]):
    """Flat name → callable map for `--sources` selection.

    The keys here are the public source names that workflow shards pass via
    `--sources foo,bar,baz`. Per-source defaults are tuned for a single
    iteration of the keep-alive loop (5-min wall-clock cap), so caller-side
    sharding doesn't need to second-guess `max_pages` etc.
    """
    return {
        # Federal-grade orgs — static HTML, low anti-bot risk.
        'ru_federal_hotlines': lambda: scrape_ru_federal_hotlines(scraper),
        'ru_banks':            lambda: scrape_ru_banks(scraper),
        'ru_telecom':          lambda: scrape_ru_telecom(scraper),
        'ru_airlines':         lambda: scrape_ru_airlines(scraper),
        'ru_airports':         lambda: scrape_ru_airports(scraper),
        # Civic / commercial orgs.
        'ru_press':            lambda: scrape_ru_press(scraper),
        'ru_universities':     lambda: scrape_ru_universities(scraper),
        'ru_insurance':        lambda: scrape_ru_insurance(scraper),
        'ru_retail':           lambda: scrape_ru_retail(scraper),
        'ru_marketplaces':     lambda: scrape_ru_marketplaces(scraper),
        'rusprofile':          lambda: scrape_rusprofile(scraper, RUSPROFILE_QUERIES),
        # State portals.
        'mosgorzdrav':         lambda: scrape_mosgorzdrav(scraper),
        'mos_ru':              lambda: scrape_mos_ru(scraper),
        # Catalog aggregators (high-yield).
        'orgpage':             lambda: scrape_orgpage(scraper, max_ranges=14, max_companies_per_range=60),
        'list_org':            lambda: scrape_list_org(scraper, max_subsitemaps=5, max_companies=300),
        'two_gis_api':         lambda: scrape_2gis_api(scraper, max_regions=8, max_queries_per_region=8, max_pages_per_query=6),
        # City directories.
        'spravker':            lambda: scrape_spravker(scraper, spravker_cities, SPRAVKER_SUBCATEGORIES),
        'zoon':                lambda: scrape_zoon(scraper, zoon_cities, ZOON_CATEGORIES),
        'zoon_smart':          lambda: scrape_zoon(scraper, zoon_cities, SMART_ZOON_CATEGORIES),
        'spravker_smart':      lambda: scrape_spravker(scraper, spravker_cities, SMART_SPRAVKER_SUBCATEGORIES),
        # Weak / private listings (anti-bot risk — keep concurrency low).
        'cian':                lambda: scrape_cian(scraper),
        'fl_ru':               lambda: scrape_fl_ru(scraper),
        'hands_ru':            lambda: scrape_hands_ru(scraper),
        'freelance_ru':        lambda: scrape_freelance_ru(scraper),
        'drom':                lambda: scrape_drom(scraper),
        'delivery_public':     lambda: scrape_delivery_public(scraper),
        'service_marketplace': lambda: scrape_service_marketplaces(scraper),
        'service_marketplace_fast': lambda: scrape_service_marketplaces_fast(scraper),
        'classified_public':   lambda: scrape_classified_public(scraper),
    }


async def run_all(scraper: AsyncScraper, spravker_cities: Dict[str, str],
                  zoon_cities: Dict[str, str], profile: str = 'smart',
                  selected_sources: Optional[List[str]] = None):
    if selected_sources:
        registry = build_source_registry(scraper, spravker_cities, zoon_cities)
        sources = []
        unknown = []
        for name in selected_sources:
            if name in registry:
                sources.append((name, registry[name]))
            else:
                unknown.append(name)
        if unknown:
            log.warning(f"Unknown sources skipped: {unknown}; "
                        f"available: {sorted(registry.keys())}")
        if not sources:
            log.error("No matching sources — nothing to scrape this iter.")
            return {}

        source_stats = {}
        for name, fn in sources:
            log.info(f"▶ Source: {name}")
            t0 = time.monotonic()
            try:
                count = await fn()
            except Exception as e:
                log.error(f"  Source {name} failed: {e}")
                count = 0
            elapsed = time.monotonic() - t0
            source_stats[name] = (count, elapsed)
            log.info(f"  ✓ {name}: {count} numbers in {elapsed:.1f}s")
            save_results(scraper.results, scraper._output_path)
            save_state(scraper, getattr(scraper, '_state_path', None))
            log.info(f"  💾 Saved {len(scraper.results)} numbers | "
                     f"fetched={scraper.stats['fetched']} skipped={scraper.stats['skipped']}")
        return source_stats

    # Контактные страницы крупных РФ-организаций (банки, телеком, авиа, аэропорты,
    # СМИ, вузы, страховые, ритейл, маркетплейсы, федеральные горячие линии).
    # Дёшевые быстрые источники (~80 URL суммарно), участвуют во всех профилях
    # для усиления ALLOW-стороны датасета.
    ru_org_sources = [
        ('ru_federal_hotlines',  lambda: scrape_ru_federal_hotlines(scraper)),
        ('ru_banks',             lambda: scrape_ru_banks(scraper)),
        ('ru_telecom',           lambda: scrape_ru_telecom(scraper)),
        ('ru_airlines',          lambda: scrape_ru_airlines(scraper)),
        ('ru_airports',          lambda: scrape_ru_airports(scraper)),
        ('ru_press',             lambda: scrape_ru_press(scraper)),
        ('ru_universities',      lambda: scrape_ru_universities(scraper)),
        ('ru_insurance',         lambda: scrape_ru_insurance(scraper)),
        ('ru_retail',            lambda: scrape_ru_retail(scraper)),
        ('ru_marketplaces',      lambda: scrape_ru_marketplaces(scraper)),
    ]

    if profile == 'weak':
        sources = [
            ('orgpage',              lambda: scrape_orgpage(scraper, max_ranges=4, max_companies_per_range=15)),
            ('delivery_public',      lambda: scrape_delivery_public(scraper)),
            ('service_marketplace_fast',  lambda: scrape_service_marketplaces_fast(scraper)),
            ('classified_public',    lambda: scrape_classified_public(scraper)),
            ('cian',                 lambda: scrape_cian(scraper, max_pages=20)),
            ('hands_ru',             lambda: scrape_hands_ru(scraper)),
            ('freelance_ru',         lambda: scrape_freelance_ru(scraper)),
            ('fl_ru',                lambda: scrape_fl_ru(scraper)),
            ('drom',                 lambda: scrape_drom(scraper)),
        ]
        sources = ru_org_sources + sources
    elif profile == 'org':
        sources = [
            ('orgpage',     lambda: scrape_orgpage(scraper, max_ranges=8, max_companies_per_range=30)),
            ('list_org',    lambda: scrape_list_org(scraper, max_subsitemaps=4, max_companies=250)),
            ('two_gis_api', lambda: scrape_2gis_api(scraper, max_regions=6, max_queries_per_region=6, max_pages_per_query=5)),
            ('zoon',        lambda: scrape_zoon(scraper, zoon_cities, ZOON_CATEGORIES)),
            ('spravker',    lambda: scrape_spravker(scraper, spravker_cities, SPRAVKER_SUBCATEGORIES)),
            ('rusprofile',  lambda: scrape_rusprofile(scraper, RUSPROFILE_QUERIES)),
            ('mosgorzdrav', lambda: scrape_mosgorzdrav(scraper)),
            ('mos_ru',      lambda: scrape_mos_ru(scraper)),
        ]
        sources = ru_org_sources + sources
    elif profile == 'spravker_only':
        sources = [
            ('spravker',    lambda: scrape_spravker(scraper, spravker_cities, SPRAVKER_SUBCATEGORIES)),
        ]
        # No ru_org_sources — direct to spravker for maximum signal-density on
        # new cities. Used for targeted city expansion runs.
    elif profile == 'broad':
        sources = [
            # orgpage is the highest-yield source we have (millions of company
            # pages, ~5-25 phones each) — put it first so even if later
            # scrapers stall on anti-bot, we've already captured fresh data.
            ('orgpage',              lambda: scrape_orgpage(scraper, max_ranges=10, max_companies_per_range=30)),
            ('list_org',             lambda: scrape_list_org(scraper, max_subsitemaps=5, max_companies=300)),
            ('two_gis_api',          lambda: scrape_2gis_api(scraper, max_regions=8, max_queries_per_region=8, max_pages_per_query=6)),
            ('delivery_public',      lambda: scrape_delivery_public(scraper)),
            ('service_marketplace',  lambda: scrape_service_marketplaces(scraper)),
            ('classified_public',    lambda: scrape_classified_public(scraper)),
            ('cian',                 lambda: scrape_cian(scraper)),
            ('fl_ru',                lambda: scrape_fl_ru(scraper)),
            ('hands_ru',             lambda: scrape_hands_ru(scraper)),
            ('freelance_ru',         lambda: scrape_freelance_ru(scraper)),
            ('drom',                 lambda: scrape_drom(scraper)),
            ('zoon',                 lambda: scrape_zoon(scraper, zoon_cities, ZOON_CATEGORIES)),
            ('spravker',             lambda: scrape_spravker(scraper, spravker_cities, SPRAVKER_SUBCATEGORIES)),
            ('rusprofile',           lambda: scrape_rusprofile(scraper, RUSPROFILE_QUERIES)),
            ('mosgorzdrav',          lambda: scrape_mosgorzdrav(scraper)),
            ('mos_ru',               lambda: scrape_mos_ru(scraper)),
        ]
        sources = ru_org_sources + sources
    else:
        sources = [
            ('orgpage',              lambda: scrape_orgpage(scraper, max_ranges=6, max_companies_per_range=25)),
            ('list_org',             lambda: scrape_list_org(scraper, max_subsitemaps=3, max_companies=200)),
            ('two_gis_api',          lambda: scrape_2gis_api(scraper, max_regions=4, max_queries_per_region=5, max_pages_per_query=4)),
            ('delivery_public',      lambda: scrape_delivery_public(scraper)),
            ('service_marketplace',  lambda: scrape_service_marketplaces(scraper)),
            ('classified_public',    lambda: scrape_classified_public(scraper)),
            ('cian',                 lambda: scrape_cian(scraper, max_pages=8)),
            ('hands_ru',             lambda: scrape_hands_ru(scraper)),
            ('freelance_ru',         lambda: scrape_freelance_ru(scraper)),
            ('fl_ru',                lambda: scrape_fl_ru(scraper)),
            ('drom',                 lambda: scrape_drom(scraper)),
            ('zoon_smart',           lambda: scrape_zoon(scraper, zoon_cities, SMART_ZOON_CATEGORIES)),
            ('spravker_smart',       lambda: scrape_spravker(scraper, spravker_cities, SMART_SPRAVKER_SUBCATEGORIES)),
            ('mos_ru',               lambda: scrape_mos_ru(scraper)),
        ]
        sources = ru_org_sources + sources

    source_stats = {}
    for name, fn in sources:
        log.info(f"▶ Source: {name}")
        t0 = time.monotonic()
        try:
            count = await fn()
        except Exception as e:
            log.error(f"  Source {name} failed: {e}")
            count = 0
        elapsed = time.monotonic() - t0
        source_stats[name] = (count, elapsed)
        log.info(f"  ✓ {name}: {count} numbers in {elapsed:.1f}s")

        # Incremental save after each source
        save_results(scraper.results, scraper._output_path)
        save_state(scraper, getattr(scraper, '_state_path', None))
        log.info(f"  💾 Saved {len(scraper.results)} numbers | fetched={scraper.stats['fetched']} skipped={scraper.stats['skipped']}")

    return source_stats


def save_results(results: List[LegitEntry], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Write to temp file first, then rename (atomic — no data loss on crash)
    tmp_path = output_path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['normalized_number', 'name', 'category', 'source', 'city', 'url', 'source_confidence'])
        for entry in sorted(results, key=lambda e: (e.category, e.normalized_number)):
            writer.writerow([entry.normalized_number, entry.name, entry.category,
                             entry.source, entry.city, entry.url, f'{entry.source_confidence:.2f}'])
    # Atomic rename
    if os.path.exists(output_path):
        os.replace(tmp_path, output_path)
    else:
        os.rename(tmp_path, output_path)
    return len(results)


# visited_urls entries older than this are dropped on next load; keeps URL
# dedup useful within a single 5.5h workflow run while still letting
# discovery pages (city listings, sitemaps) be re-walked across waves.
VISITED_TTL_S = 6 * 60 * 60


def save_state(scraper: AsyncScraper, state_path: Optional[str] = None):
    """Save visited URLs (with timestamps) to state file for resume."""
    state_path = state_path or STATE_PATH
    now = int(time.time())
    visited_ts = getattr(scraper, 'visited_url_ts', {})
    # Stamp any URLs missing a timestamp (legacy state) with `now` so they
    # age out from this point forward instead of being treated as fresh
    # forever.
    for url in scraper.visited_urls:
        visited_ts.setdefault(url, now)
    state = {
        'visited_url_ts': visited_ts,
        'fetched': scraper.stats['fetched'],
    }
    os.makedirs(os.path.dirname(state_path) or '.', exist_ok=True)
    tmp = state_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(state, f)
    if os.path.exists(state_path):
        os.replace(tmp, state_path)
    else:
        os.rename(tmp, state_path)


def load_state(scraper: AsyncScraper, state_path: Optional[str] = None):
    """Load visited URLs from state file, dropping entries older than TTL."""
    state_path = state_path or STATE_PATH
    if not os.path.exists(state_path):
        return
    with open(state_path, encoding='utf-8') as f:
        state = json.load(f)
    cutoff = int(time.time()) - VISITED_TTL_S
    visited_ts: Dict[str, int] = {}
    if 'visited_url_ts' in state and isinstance(state['visited_url_ts'], dict):
        for url, ts in state['visited_url_ts'].items():
            try:
                if int(ts) >= cutoff:
                    visited_ts[url] = int(ts)
            except (TypeError, ValueError):
                continue
    else:
        # Backwards-compat: legacy state had `visited_urls: List[str]`
        # without timestamps. Treat them as fresh-now so the TTL takes
        # effect from this load forward.
        now = int(time.time())
        for url in state.get('visited_urls', []) or []:
            visited_ts[url] = now
    scraper.visited_urls = set(visited_ts.keys())
    scraper.visited_url_ts = visited_ts
    scraper.stats['fetched'] = state.get('fetched', 0)
    scraper._resumed_fetched = scraper.stats['fetched']
    log.info(f"  Resumed state: {len(scraper.visited_urls)} visited URLs (after TTL purge), "
             f"{scraper.stats['fetched']} fetched")


async def main():
    parser = argparse.ArgumentParser(description='Сбор легитимных номеров РФ')
    parser.add_argument('--cities', nargs='+', default=['msk', 'spb', 'ekb', 'kzn', 'nnov', 'rnd', 'ufa', 'krasnodar', 'voronezh', 'chelyabinsk'],
                        help='Города (ключи: msk, spb, ekb, kzn, ...)'),
    parser.add_argument('--profile', choices=['smart', 'broad', 'org', 'weak', 'spravker_only'], default='smart',
                        help='smart=сначала слабые/потом org, broad=всё, org=только организации, weak=только слабые категории')
    parser.add_argument('--concurrency', type=int, default=CONCURRENCY,
                        help='Макс. одновременных HTTP-запросов')
    parser.add_argument('--max-urls', type=int, default=0,
                        help='Макс. HTTP-запросов (0=без лимита)')
    parser.add_argument('--add-user-numbers', type=int, default=0,
                        help='Добавить N низкоуверенных обычных мобильных номеров из официального плана нумерации')
    parser.add_argument('--output', default=OUTPUT_PATH,
                        help='Выходной CSV файл')
    parser.add_argument('--state-path', default='',
                        help='Путь к JSON-файлу состояния (по умолчанию рядом с --output)')
    parser.add_argument('--sources', default='',
                        help='Запустить только перечисленные через запятую источники '
                             '(имена из build_source_registry); --profile игнорируется. '
                             'Пример: --sources ru_banks,ru_telecom,orgpage')
    args = parser.parse_args()

    # Phase-1 ALLOW ×10: '--cities all' is shorthand for "use every city
    # registered in SPRAVKER_CITIES + ZOON_CITIES". Avoids having to keep a
    # 130-city list inline in the workflow.
    if len(args.cities) == 1 and args.cities[0].lower() == 'all':
        args.cities = sorted(set(SPRAVKER_CITIES) | set(ZOON_CITIES))

    spravker_cities = {k: v for k, v in SPRAVKER_CITIES.items() if k in args.cities}
    zoon_cities = {k: v for k, v in ZOON_CITIES.items() if k in args.cities}

    log.info(f"🚀 Starting legitimate number collector")
    log.info(f"  Cities: {len(args.cities)} ({', '.join(args.cities[:8])}{'…' if len(args.cities) > 8 else ''})")
    log.info(f"  Spravker cities: {len(spravker_cities)} ({', '.join(list(spravker_cities)[:6])}…)")
    log.info(f"  Zoon cities: {len(zoon_cities)} ({', '.join(list(zoon_cities)[:6])}…)")
    log.info(f"  Profile: {args.profile}")
    log.info(f"  Concurrency: {args.concurrency}")

    state_path = args.state_path or os.path.join(
        os.path.dirname(os.path.abspath(args.output)) or '.',
        'legitimate_collector_state.json',
    )
    log.info(f"  State path: {state_path}")

    selected_sources: Optional[List[str]] = None
    if args.sources.strip():
        selected_sources = [s.strip() for s in args.sources.split(',') if s.strip()]
        log.info(f"  Sources override: {selected_sources}")

    scraper = AsyncScraper(concurrency=args.concurrency)
    scraper._output_path = args.output  # for incremental saves
    scraper._state_path = state_path    # per-shard state file
    scraper._max_urls = args.max_urls   # 0 = unlimited
    await scraper.start()

    try:
        # 0a. Resume state (visited URLs)
        load_state(scraper, state_path)

        # 0b. Resume: load existing CSV so data isn't lost on restart
        if os.path.exists(args.output):
            with open(args.output, encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    num = row.get('normalized_number', '')
                    if num and num not in scraper.seen:
                        scraper.seen.add(num)
                        scraper.results.append(LegitEntry(
                            num,
                            row.get('name', ''),
                            row.get('category', ''),
                            row.get('source', ''),
                            row.get('city', ''),
                            row.get('url', ''),
                            parse_confidence(row.get('source_confidence'), 0.70),
                        ))
            log.info(f"  Resumed {len(scraper.results)} existing entries from {args.output}")

        # 1. Load existing whitelist
        official = load_official_whitelist()
        for entry in official:
            scraper.add(entry)
        log.info(f"  Loaded {len(official)} official whitelist entries")

        # 2. Add hardcoded hotlines
        for num, name, cat in OFFICIAL_HOTLINES:
            scraper.add(LegitEntry(num, name, cat, 'official_hotline', '', '', 0.95))
        log.info(f"  Added {len(OFFICIAL_HOTLINES)} hardcoded hotlines")

        # 3. Load blacklist (fraud/suspect numbers to exclude)
        scraper.load_blacklist()

        # 4. Scrape all sources
        stats = await run_all(scraper, spravker_cities, zoon_cities,
                              profile=args.profile,
                              selected_sources=selected_sources)

        if args.add_user_numbers > 0:
            log.info(f"▶ Source: numbering_plan_background ({args.add_user_numbers})")
            t0 = time.monotonic()
            count = generate_user_numbers(scraper, args.add_user_numbers)
            stats['numbering_plan_background'] = (count, time.monotonic() - t0)
            save_results(scraper.results, scraper._output_path)
            log.info(f"  💾 Saved {len(scraper.results)} numbers after background users")

        # 4. Save
        total = save_results(scraper.results, args.output)
        log.info(f"\n{'='*60}")
        log.info(f"✅ Done! {total} unique legitimate numbers saved to {args.output}")
        log.info(f"   HTTP: {scraper.stats['fetched']} fetched, {scraper.stats['failed']} failed")
        log.info(f"\n   Source breakdown:")
        for name, (count, elapsed) in sorted(stats.items(), key=lambda x: -x[1][0]):
            log.info(f"     {name:20s}: {count:5d} numbers ({elapsed:.1f}s)")

        cats: Dict[str, int] = {}
        for e in scraper.results:
            cats[e.category] = cats.get(e.category, 0) + 1
        log.info(f"\n   Category breakdown:")
        for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
            log.info(f"     {cat:20s}: {count:5d}")

    finally:
        # Always save what we have
        save_results(scraper.results, args.output)
        save_state(scraper, state_path)
        log.info(f"💾 Final save: {len(scraper.results)} numbers | {len(scraper.visited_urls)} URLs visited")
        await scraper.close()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠ Interrupted — data already saved incrementally")

# restart-trigger 2026-04-29T20:45 — keep-alive workflows died at 19:22, push to retrigger
