package Test;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

@SpringBootTest
class SpringbootSchemaApplicationTests {

	@Test
	void Test() throws ClassNotFoundException, NoSuchFieldException, IllegalAccessException, NoSuchMethodException, InvocationTargetException {
		// 通过 Class.forName 获取 Class 对象
		Class<?> clazz = Class.forName("com.example.Person");

		// 获取字段并操作
		Field nameField = clazz.getDeclaredField("name");
		nameField.setAccessible(true);  // 设置访问私有字段
		Person person = new Person("John", 25);
		System.out.println("Name: " + nameField.get(person));  // 获取字段值

		// 修改字段值
		nameField.set(person, "Alice");
		System.out.println("Updated Name: " + nameField.get(person));

		// 获取方法并调用
		Method greetMethod = clazz.getDeclaredMethod("greet");
		greetMethod.setAccessible(true);  // 设置访问私有方法
		greetMethod.invoke(person);  // 调用 greet 方法
	}

	public static class Person {
		private String name;
		private  int age;
		public Person(String name,int age){
			this.name = name;
			this.age = age;
		}
	}

}
