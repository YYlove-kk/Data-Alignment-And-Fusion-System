const menu = {
    list() {
        return [
            {
            "backMenu": [
                {
                "child": [{
                    "buttons": ["新增", "查看", "修改", "删除"],
                    "menu": "设备状态",
                    "menuJump": "列表",
                    "tableName": "device"
                }],
                "menu": "设备状态查看"
                },
                {
                    "child": [{
                        "buttons": ["新增", "查看", "修改", "删除"],
                        "menu": "模组状态",
                        "tableName": "BatteryModule"
                    }],
                    "menu": "模组状态查看"
                },
                {
                    "child": [{
                        "buttons": ["新增", "查看", "修改", "删除"],
                        "menu": "异常模组状态",
                        "tableName": "ExceptionBatteryModule"
                    }],
                    "menu": "异常状态记录管理"
                },
                {
                    "child": [{
                        "buttons": ["新增", "查看", "修改", "删除"],
                        "menu": "分区管理",
                        "tableName": "DivideControl"
                    }],
                    "menu": "分区管理"
                },
                {
                    "child": [{
                        "buttons": ["新增", "查看", "修改", "删除"],
                        "menu": "未处理模组",
                        "tableName": "Stock"
                    }],
                    "menu": "未处理模组"
                }
            ],
                "roleName": "管理员",
                "tableName": "admin"
            },

            {
                "backMenu": [
                    {
                        "child": [{
                            "buttons": ["新增", "查看", "修改", "删除"],
                            "menu": "设备状态",
                            "menuJump": "列表",
                            "tableName": "device"
                        }],
                        "menu": "设备状态查看"
                    },
                    {
                        "child": [{
                            "buttons": ["新增", "查看", "修改", "删除"],
                            "menu": "模组状态",
                            "tableName": "BatteryModule"
                        }],
                        "menu": "模组状态查看"
                    },
                    {
                        "child": [{
                            "buttons": ["新增", "查看", "修改", "删除"],
                            "menu": "异常模组",
                            "tableName": "ExceptionBatteryModule"
                        }],
                        "menu": "异常状态记录管理"
                    }
                ],
                "roleName": "用户",
                "tableName": "users"
            }

        ]
    }
}
export default menu;
