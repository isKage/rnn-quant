import baostock as bs


def stock_name(code):
    # 登陆系统
    # bs.login()

    # 获取证券基本资料
    rs = bs.query_stock_basic(code=code)

    # 打印结果集
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())

    # 登出系统
    # bs.logout()

    return data_list[0][1]


def stock_code(code_name):
    # 登陆系统
    # bs.login()

    # 获取证券基本资料
    rs = bs.query_stock_basic(code_name=code_name)

    # 打印结果集
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())

    # 登出系统
    # bs.logout()

    return data_list[0][0]

if __name__ == '__main__':
    bs.login()

    code = 'sh.600036'
    code_name = '比亚迪'
    print(stock_name(code))
    print(stock_code(code_name))

    bs.logout()